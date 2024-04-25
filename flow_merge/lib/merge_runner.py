import argparse
from typing import Union

import torch

from flow_merge.lib.architecture import ModelArchitecture, get_all_weights
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import MergeConfig
from flow_merge.lib.merger import Merger
from flow_merge.lib.tensor_loader import TensorIndex, TensorLoader
from flow_merge.lib.tensor_writer import TensorWriter
from flow_merge.lib.tokenizer import get_merge_tokenizer
from flow_merge.lib.utils import generate_model_card

logger = get_logger(__name__)


def validate_architectures(merge_config: MergeConfig) -> ModelArchitecture:
    """
    Validate that all architectures are the same and therefore compatible.
    """
    base_model_arch = ModelArchitecture.from_config(merge_config.base_model.config)
    model_archs = [
        ModelArchitecture.from_config(model.config) for model in merge_config.models
    ]

    if not all(
        set(base_model_arch.architectures).intersection(set(model_arch.architectures))
        for model_arch in model_archs
    ):
        raise RuntimeError(
            f"You are trying to merge models with different architectures. This is not supported."
        )

    if not all(
        base_model_arch.weights == model_arch.weights for model_arch in model_archs
    ):
        raise RuntimeError(
            f"You are trying to merge models with different weights. This is not supported."
        )

    if not all(
        base_model_arch.model_type == model_arch.model_type
        for model_arch in model_archs
    ):
        raise RuntimeError(
            f"You are trying to merge models with different architectures. This is not supported."
        )

    return base_model_arch


def run_merge(config: dict | argparse.Namespace, model_name: str = "Untitled") -> None:
    """
    Merges multiple models into a single model based on the provided configuration.
    Contains the logic for the merge process.

    Arguments:
        config: Configuration for the merge operation.
            - If config is an argparse.Namespace, it should contain the following attributes:
                - config (str): Path to the YAML configuration file.
                - model_name (str): Name of the resulting merged model.
            - If config is a MergeConfig object, it will be used directly.

    Usage:
        # Using a YAML configuration file
        python run_merge.py --config /path/to/merge_config.yaml --model_name my_merged_model

        # Using a MergeConfig object
        merge_config = MergeConfig(...)
        run_merge(merge_config)
    """
    try:
        logger.info("Starting merge...")
        if isinstance(config, argparse.Namespace):
            # If config is an argparse Namespace, read the merge config from the specified file
            merge_config = MergeConfig.from_yaml(config.config)
            model_name = config.model_name if config.model_name else "Untitled"
        elif isinstance(config, dict):
            # If config is a dict object, use it directly
            merge_config = MergeConfig.from_dict(config)
        else:
            TypeError(
                "Input to run_merge needs to be either a string path to a YAML config file or a dict"
            )

        # * Architecture validation
        model_arch = validate_architectures(merge_config)

        # Tokenizer
        tokenizer = get_merge_tokenizer(merge_config)

        # Tensor loaders - {Model: TensorLoader, ...}
        tensor_indices = {
            model: TensorIndex(str(model.path), merge_config)
            for model in merge_config.models + [merge_config.base_model]
        }
        tensor_loaders = {
            model: TensorLoader(tensor_indices[model], merge_config)
            for model in merge_config.models + [merge_config.base_model]
        }

        # Initialize merger
        merger = Merger(
            merge_config=merge_config,
            tensor_loaders=tensor_loaders,
            input_ids_mappings=tokenizer.input_ids_mappings,  # ! NOTE - With input_ids_mappings
        )

        # Initialize writer
        with TensorWriter(merge_config=merge_config) as writer:
            for weight in get_all_weights(model_arch):
                if tokenizer.input_ids_mappings and (
                    weight.layer_type == "embedding" or weight.layer_type == "head"
                ):
                    merged_tensor = merger.interpolate(weight)
                else:
                    merged_tensor = merger.merge_weights(weight)
                writer.save_tensor(weight=weight, tensor=merged_tensor, clone=False)
            writer.finish()

        # Merge model config
        if merge_config.base_model:
            merged_model_config = merge_config.base_model.config
            # merged_model_dtype = merge_config.base_model.model.dtype
        else:
            merged_model_config = merge_config.models[0].config
            # merged_model_dtype = merge_config.models[0].model.dtype

        # Set _name_or_path to local output dir
        merged_model_config._name_or_path = merge_config.directory_settings.output_dir

        # ! If input id mappings rectify the embeeding size, the vocab_size isn't correct in the base model config since it uses the num of embeddings...
        # Update vocab size
        if tokenizer.input_ids_mappings:
            merged_model_config.vocab_size = len(tokenizer.tokenizer.get_vocab())

        # save tokenizer and config
        logger.info(f"Saving tokenizer to {merge_config.directory_settings.output_dir}")
        tokenizer.tokenizer.save_pretrained(
            merge_config.directory_settings.output_dir, safe_serialization=True
        )
        logger.info(
            f"Saving config.json to {merge_config.directory_settings.output_dir}"
        )
        merged_model_config.save_pretrained(merge_config.directory_settings.output_dir)
        logger.info(
            f"Saving merge config to {merge_config.directory_settings.output_dir}"
        )
        merge_config.save_config()

        # Generate model card
        generate_model_card(merge_config, model_name)
        logger.info("Merge completed.")

    except Exception as e:
        logger.error(f"Merge error: {type(e).__name__} - {str(e)}")
