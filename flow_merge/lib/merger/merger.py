import logging
from typing import List, Tuple
import torch
from transformers import AutoConfig
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.merge_methods import MergeMethodIdentifier, TaskArithmetic, TiesMergingSettings, \
    DareTiesMergingSettings, TaskArithmeticSettings
from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.interpolation import interpolate
from flow_merge.lib.merge_methods.passthrough import merge_passthrough
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.model.architecture import ModelWeight, ModelArchitectureProvider
from flow_merge.lib.model.metadata import ModelMetadataService
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tensor.writer import TensorWriter
from flow_merge.lib.tokenizer import MergeTokenizerService
from flow_merge.lib.hf.upload import generate_model_card

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_merge_methods = {
    MergeMethodIdentifier.PASSTHROUGH: merge_passthrough,
    MergeMethodIdentifier.MODEL_SOUP: merge_linear,
    MergeMethodIdentifier.SLERP: merge_slerp,
    MergeMethodIdentifier.TIES_MERGING: TaskArithmetic.merge,
    MergeMethodIdentifier.DARE_TIES_MERGING: TaskArithmetic.merge,
    MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC: TaskArithmetic.merge,
}


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

        # Don't count the embedding layer, the LM head layer
        # This should be the hidden_num_layers
        hidden_dim = max(merge_plan.slices, key=lambda x: x.output_layer_id).output_layer_id + 1

        with TensorWriter(output_dir=self.config.output_dir) as writer:
            for idx, s in enumerate(merge_plan.slices):

                logger.debug(f"Merging slice {idx}")
                # Fixme: creating map of all models to their weights (layers names)
                tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool, str]] = []
                for source in s.sources:
                    metadata = self.metadata_service.load_model_metadata(source.model)
                    shards = self.model_service.create_shard_files(model_metadata=metadata)

                    if source.is_base:
                        merged_model_config = metadata.config

                    tensor = self.tensor_repository.get_tensor(
                        shards=shards,
                        tensor_key=self.get_model_weight(source.model, source.layer).name,
                        device=self.config.device,
                    )
                    tensors_weights_pairs.append((
                        tensor, 
                        source.weight, 
                        source.is_base, 
                        source.model,
                        source.layer
                        ))

                if tokenizer.input_ids_mappings and s.merge_method.name == MergeMethodIdentifier.INTERPOLATE:

                    writer.save_tensor(
                        weight_name=s.output_layer_name, 
                        tensor=interpolate(
                        all_tensors=tensors_weights_pairs,
                        input_ids_mappings=tokenizer.input_ids_mappings,
                        merge_method_name=s.merge_method.name,
                    ))
                    continue
                elif s.merge_method.name == MergeMethodIdentifier.INTERPOLATE:
                    continue

                merge_alg_settings = {}
                if s.merge_method.name == MergeMethodIdentifier.MODEL_SOUP:
                    merge_alg_settings = {**s.merge_method.params}

                if s.merge_method.name == MergeMethodIdentifier.SLERP:
                    merge_alg_settings = SlerpSettings(**(s.merge_method.params or {}))

                if (s.merge_method.name in [MergeMethodIdentifier.TIES_MERGING,
                                            s.merge_method.name == MergeMethodIdentifier.DARE_TIES_MERGING,
                                            s.merge_method.name == MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC
                                            ]):
                    settings_class = {
                        MergeMethodIdentifier.TIES_MERGING: TiesMergingSettings,
                        MergeMethodIdentifier.DARE_TIES_MERGING: DareTiesMergingSettings,
                        MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC: TaskArithmeticSettings,
                    }
                    merge_alg_settings = settings_class[s.merge_method.name](**s.merge_method.params)

                writer.save_tensor(
                    weight_name=s.output_layer_name, 
                    tensor=_merge_methods[s.merge_method.name](
                    tensors_weights_pairs=tensors_weights_pairs,
                    merge_method_settings=merge_alg_settings,
                ))
            # clean-up
            writer.finish()

        merged_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.local_dir / merge_plan.base_model,
            trust_remote_code=self.config.trust_remote_code
        )

        merged_config._name_or_path = str(self.config.output_dir)

        merged_config.num_hidden_layers = hidden_dim

        if tokenizer.input_ids_mappings:
            merged_config.vocab_size = len(tokenizer.tokenizer.get_vocab())

        logger.info(f"Saving tokenizer to {self.config.output_dir}")
        
        tokenizer.tokenizer.save_pretrained(
            self.config.output_dir, safe_serialization=True
        )

        logger.info(
            f"Saving config.json to {self.config.output_dir}"
        )

        merged_config.save_pretrained(self.config.output_dir)

        generate_model_card(merge_plan=merge_plan, app_config=self.config)


