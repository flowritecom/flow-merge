import logging
from typing import List, Tuple

import torch

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.loaders.normalizer import NormalizedSource
from flow_merge.lib.merge_methods import MergeMethodIdentifier, TaskArithmetic, TiesMergingSettings, \
    DareTiesMergingSettings, TaskArithmeticSettings
from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.model import Model
from flow_merge.lib.model.architecture import ModelArchitecture, ModelWeight
from flow_merge.lib.model.metadata import ModelMetadataService
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tokenizer import MergeTokenizerService
from flow_merge.lib.merger.interpolation import InterpolationRunner

config = ApplicationConfig()


def get_model_weight(model_path_or_id: str, weight_name: str) -> ModelWeight:
    arch = ModelArchitecture.from_path_or_id(model_path_or_id, config.local_dir, env=config)
    return arch.get_weight(weight_name)


def get_base_source(sources: List[NormalizedSource]) -> NormalizedSource:
    return [src for src in sources if src.is_base is True][0]


def merge(
        merge_plan: MergePlan
):
    logging.basicConfig(level=logging.INFO)
    tokenizer_service = MergeTokenizerService(config=config)
    tokenizer = tokenizer_service.get_merge_tokenizer(merge_plan)
    metadata_service = ModelMetadataService(app_config=config)

    output = []

    for idx, s in enumerate(merge_plan.slices):
        print(f"Merging slice {idx}")
        # Fixme: creating map of all models to their weights (layers names)
        tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool]] = []
        for source in s.sources:
            metadata = metadata_service.load_model_metadata(source.model)
            shards = ModelService.create_shard_files(model_metadata=metadata, app_config=config)

            tensor = TensorRepository.get_tensor(
                shards=shards,
                tensor_key=get_model_weight(source.model, source.layer).name,
                device=config.device,
            )
            tensors_weights_pairs.append((tensor, source.weight, source.is_base))

        # hidden_dim = self._validate_tensor_shapes(
        #     base_model_weight=task_base_model_weight,
        #     tensors=all_tensors,
        #     base_model_layer_type=task_base_model_weight.layer_type
        # )

        # FIXME we want to temp save here
        if tokenizer.input_ids_mappings and s.merge_method.name == MergeMethodIdentifier.INTERPOLATE:
            interpolation_runner = InterpolationRunner
            output.append( interpolation_runner.interpolate(
                all_tensors=tensors_weights_pairs,
                merge_method_name=s.merge_method.name,
                input_ids_mappings=tokenizer.input_ids_mappings,
                hidden_dim=1  # fixme: hidden dimensions are unknown at this point
            ))
            continue

        if s.merge_method.name == MergeMethodIdentifier.MODEL_SOUP:
            output.append(merge_linear(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings={"normalize": s.merge_method.params["normalize"] or False}
            ))
            continue

        if s.merge_method.name == MergeMethodIdentifier.SLERP:
            output.append( merge_slerp(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings=SlerpSettings(**(s.merge_method.params or {})),
            ))
            continue

        if (s.merge_method.name in [MergeMethodIdentifier.TIES_MERGING,
                                    s.merge_method.name == MergeMethodIdentifier.DARE_TIES_MERGING,
                                    s.merge_method.name == MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC
                                    ]):
            task_arithmetic_merger = TaskArithmetic()  # fixme: for now an object instance, let's see if needed later
            settings_class = {
                MergeMethodIdentifier.TIES_MERGING: TiesMergingSettings,
                MergeMethodIdentifier.DARE_TIES_MERGING: DareTiesMergingSettings,
                MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC: TaskArithmeticSettings,
            }

            output.append( task_arithmetic_merger.merge(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings=settings_class[s.merge_method.name](**s.merge_method.params),
            ))
            continue

    print(output)
