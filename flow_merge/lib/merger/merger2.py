from typing import List, Tuple

import torch

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.merge_methods import MergeMethodIdentifier, TaskArithmetic, TiesMergingSettings, \
    DareTiesMergingSettings, TaskArithmeticSettings
from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.model.architecture import ModelArchitecture, ModelWeight
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tokenizer import Tokenizer
from flow_merge.lib.merger.interpolation import InterpolationRunner

config = ApplicationConfig()


def get_model_weight(model_path_or_id: str, weight_name: str) -> ModelWeight:
    arch = ModelArchitecture.from_path_or_id(model_path_or_id, config.local_dir, env=config)
    return arch.get_weight(weight_name)


def get_base_source(sources: List[NormalizedSource]) -> NormalizedSource:
    return [src for src in sources if src.is_base is True][0]


def merge(
        self,
        tokenizer: Tokenizer,
        merge_plan: MergePlan
):
    for s in merge_plan.slices:
        # Fixme: creating map of all models to their weights (layers names)
        tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool]] = []
        for source in s.sources:
            tensor = TensorRepository.get_tensor(
                shards=source.model.shards,
                tensor_key=get_model_weight(source.model, source.layer).name,
                device=self.env.device
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
            return interpolation_runner.interpolate(
                all_tensors=tensors_weights_pairs,
                merge_method_name=s.merge_method.name,
                input_ids_mappings=tokenizer.input_ids_mappings,
                hidden_dim=hidden_dim
            )

        if s.merge_method.name == MergeMethodIdentifier.MODEL_SOUP:
            return merge_linear(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings={"normalize": s.merge_method.params["normalize"] or False}
            )

        if s.merge_method.name == MergeMethodIdentifier.SLERP:
            return merge_slerp(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings=SlerpSettings(**s.merge_method.params),
            )

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

            return task_arithmetic_merger.merge(
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings=settings_class[s.merge_method.name](**s.merge_method.params),
            )

        # return method_config.method.merge(
        #     weight=task_base_model_weight,                  # who knows why it's here
        #     base_model=base_model,                          # base model STRING NAME
        #     base_model_tensor=base_model_tensor,            # base model tensor, special case
        #     models_tensors=models_tensors,                  # map of all models to their tensors
        #     merge_method_settings=s.merge_method.params,    # merge method settings
        # )
