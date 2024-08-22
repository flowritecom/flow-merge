import logging
from typing import List, Tuple
import torch
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.merge_methods import MergeMethodIdentifier, TaskArithmetic, TiesMergingSettings, \
    DareTiesMergingSettings, TaskArithmeticSettings
from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.model.architecture import ModelWeight, ModelArchitectureProvider
from flow_merge.lib.model.metadata import ModelMetadataService
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tokenizer import MergeTokenizerService
from flow_merge.lib.merger.interpolation import InterpolationRunner

logger = logging.getLogger(__name__)


class Merger:

    def __init__(
            self,
            config: ApplicationConfig,
            tokenizer_service: MergeTokenizerService,
            metadata_service: ModelMetadataService,
            model_service: ModelService,
            model_arch_provider: ModelArchitectureProvider,
            tensor_repository: TensorRepository,
    ):
        self.config = config
        self.tokenizer_service = tokenizer_service
        self.metadata_service = metadata_service
        self.model_service = model_service
        self.model_arch_provider = model_arch_provider
        self.tensor_repository = tensor_repository

    def get_model_weight(self, model_path_or_id: str, weight_name: str) -> ModelWeight:
        arch = self.model_arch_provider.get_by_id(model_path_or_id)
        return arch.get_weight(weight_name)

    def execute(
            self,
            merge_plan: MergePlan,
    ):
        tokenizer = self.tokenizer_service.get_merge_tokenizer(merge_plan)

        output = []

        for idx, s in enumerate(merge_plan.slices):
            logger.debug(f"Merging slice {idx}")
            # Fixme: creating map of all models to their weights (layers names)
            tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool]] = []
            for source in s.sources:
                metadata = self.metadata_service.load_model_metadata(source.model)
                shards = self.model_service.create_shard_files(model_metadata=metadata)

                tensor = self.tensor_repository.get_tensor(
                    shards=shards,
                    tensor_key=self.get_model_weight(source.model, source.layer).name,
                    device=self.config.device,
                )
                tensors_weights_pairs.append((tensor, source.weight, source.is_base))

            # hidden_dim = self._validate_tensor_shapes(
            #     base_model_weight=task_base_model_weight,
            #     tensors=all_tensors,
            #     base_model_layer_type=task_base_model_weight.layer_type
            # )

            # FIXME we want to temp save here
            if tokenizer.input_ids_mappings and s.merge_method.name == MergeMethodIdentifier.INTERPOLATE:
                hidden_dim = max(merge_plan.slices, key=lambda x: x.output_layer_id).output_layer_id + 1
                output.append(InterpolationRunner.interpolate(
                    all_tensors=tensors_weights_pairs,
                    merge_method_name=s.merge_method.name,
                    input_ids_mappings=tokenizer.input_ids_mappings,
                    hidden_dim=hidden_dim
                ))
                continue

            if s.merge_method.name == MergeMethodIdentifier.MODEL_SOUP:
                output.append(merge_linear(
                    tensors_weights_pairs=tensors_weights_pairs,
                    merge_method_settings={**s.merge_method.params}
                ))
                continue

            if s.merge_method.name == MergeMethodIdentifier.SLERP:
                output.append(merge_slerp(
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

                output.append(task_arithmetic_merger.merge(
                    tensors_weights_pairs=tensors_weights_pairs,
                    merge_method_settings=settings_class[s.merge_method.name](**s.merge_method.params),
                ))
                continue

        print(output)
