from typing import List

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.merge_methods import MergeMethodIdentifier, TaskArithmetic, TiesMergingSettings, \
    DareTiesMergingSettings, TaskArithmeticSettings
from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.merge_method import MergeMethod
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.model.architecture import ModelArchitecture, ModelWeight
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tokenizer import Tokenizer

config = ApplicationConfig()


def get_model_weight(model_path_or_id: str, weight_name: str) -> ModelWeight:
    arch = ModelArchitecture.from_path_or_id(model_path_or_id, config.local_dir, env=config)
    return arch.get_weight(weight_name)


def get_base_source(sources: List[NormalizedSource]) -> NormalizedSource:
    return [src for src in sources if src.is_base is True][0]


def merge(
        self,
        tokenizer: Tokenizer,
        sources,
        merge_plan: MergePlan
):
    base_model = merge_plan.base_model

    for s in merge_plan.slices:
        # Fixme: creating map of all models to their weights (layers names)
        models_tensors = {}
        tensors_weights_pairs = []
        base_model_weight = 0.0
        for source in s.sources:
            if source.is_base:
                base_model_weight = source.weight
                continue

            tensor = TensorRepository.get_tensor(
                shards=source.model.shards,
                tensor_key=get_model_weight(source.model, source.layer).name,
                device=self.env.device
            )
            models_tensors[source.model] = tensor
            tensors_weights_pairs.append((tensor, source.weight))

        # Fixme: Handling base model and it's tensor
        task_base_model_weight = get_model_weight(base_model, get_base_source(s.sources).layer)
        base_model_tensor = TensorRepository.get_tensor(
            shards=base_model.shards,
            tensor_key=task_base_model_weight.name,
            device=self.env.device
        )

        all_tensors = {
            base_model: base_model_tensor,
            **models_tensors
        }

        # hidden_dim = self._validate_tensor_shapes(
        #     base_model_weight=task_base_model_weight,
        #     tensors=all_tensors,
        #     base_model_layer_type=task_base_model_weight.layer_type
        # )

        # FIXME we want to temp save here
        if tokenizer.input_ids_mappings and method_config.name == MergeMethodIdentifier.INTERPOLATE:
            return self.interpolation_runner.interpolate(
                base_model=base_model,
                all_tensors=all_tensors,
                merge_method=method_config,
                input_ids_mappings=tokenizer.input_ids_mappings,
                sources=sources,
                hidden_dim=hidden_dim
            )

        if s.merge_method.name == MergeMethodIdentifier.MODEL_SOUP:
            return merge_linear(
                base_model_tensor=base_model_tensor,
                base_model_weight=base_model_weight,
                tensors_weights_pairs=tensors_weights_pairs,
                merge_method_settings={"normalize": s.merge_method.params["normalize"] or False}
            )

        if s.merge_method.name == MergeMethodIdentifier.SLERP:
            return merge_slerp(
                base_model_tensor=base_model_tensor,
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
                base_model_tensor=base_model_tensor,
                tensor_weight_pairs=tensors_weights_pairs,
                merge_method_settings=settings_class[s.merge_method.name](**s.merge_method.params),
            )

        # return method_config.method.merge(
        #     weight=task_base_model_weight,                  # who knows why it's here
        #     base_model=base_model,                          # base model STRING NAME
        #     base_model_tensor=base_model_tensor,            # base model tensor, special case
        #     models_tensors=models_tensors,                  # map of all models to their tensors
        #     merge_method_settings=s.merge_method.params,    # merge method settings
        # )
