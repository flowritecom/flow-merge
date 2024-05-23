import argparse

from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import MergeConfig
from flow_merge.lib.merger import Merge
from flow_merge.lib.utils import generate_model_card

logger = get_logger(__name__)


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
        merge = Merge(MergeConfig.load(config))
        merge.process_and_save_weights()

        ####################################################################################################
        ## TODO: REFACTOR 
        ####################################################################################################

        # CREATION OF THE NEW OUTPUT MODEL CONFIG
        # if no base model is provided, use the first model as base model
        # TODO: check whether it's the case at any point that we would have no base model?
        if merge_config.base_model:
            merged_model_config = merge_config.base_model.config
        else:
            merged_model_config = merge_config.models[0].config

        # Set _name_or_path to local output dir
        # FIXME: This is not a clean way as merged_model_config has "state"
        merged_model_config._name_or_path = merge_config.directory_settings.output_dir

        # FIXME: We can do this inside merge_config 
        # ! If input id mappings exist, rectify the embedding size, 
        # in that case the vocab_size isn't correct in the base model config
        # since it uses the num of embeddings
        # Update vocab size
        if merge_config.tokenizer.input_ids_mappings:
            merged_model_config.vocab_size = len(
                merge_config.tokenizer.tokenizer.get_vocab()
            )
        # FIXME: SHOULD BE OWNED BY MERGER
        # save config
        merged_model_config.save_pretrained(merge_config.directory_settings.output_dir)
        logger.info(
            f"Saving merge config to {merge_config.directory_settings.output_dir}"
        )
        # FIXME: SHOULD BE OWNED BY MERGER
        merge_config.save_config()



        # FIXME: SHOULD BE OWNED BY MERGER
        # save tokenizer 
        logger.info(f"Saving tokenizer to {merge_config.directory_settings.output_dir}")
        merge_config.tokenizer.tokenizer.save_pretrained(
            merge_config.directory_settings.output_dir, safe_serialization=True
        )
        logger.info(
            f"Saving config.json to {merge_config.directory_settings.output_dir}"
        )


        # FIXME: SHOULD BE OWNED BY MERGER
        generate_model_card(merge_config, model_name)

        logger.info("Merge completed.")
        
        ####################################################################################################
        ## TODO: REFACTOR 
        ####################################################################################################

    except Exception as e:
        logger.error(f"Merge error: {type(e).__name__} - {str(e)}")
