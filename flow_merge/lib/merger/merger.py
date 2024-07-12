import torch

from typing import Dict

from flow_merge.lib.model import Model
from flow_merge.lib.tokenizer import Tokenizer
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger
from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.merger.interpolation import InterpolationRunner
from flow_merge.lib.merge_methods import MergeMethodIdentifier
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource

class Merger:

    def __init__(
            self, 
            env: ApplicationConfig, 
            logger: Logger,
            interpolation_runner: InterpolationRunner = InterpolationRunner
        ):
        self.env = env
        self.logger = logger
        self.interpolation_runner = interpolation_runner

    def _validate_tensor_shapes(
        self,
        base_model_weight: ModelWeight,
        tensors: Dict[Model, torch.Tensor],
        base_model_layer_type: str
    ):
        hidden_size = next(iter(tensors.values())).shape[
            1 if base_model_layer_type == "embedding" else 0
        ]
        for model, tensor in tensors.items():
            current_size = tensor.shape[1] if base_model_layer_type == "embedding" else tensor.shape[0]
            if current_size != hidden_size:
                raise RuntimeError(
                    f"Tensor shape mismatch in '{base_model_weight.name}'. Expected {hidden_size}, but {model.path} has {current_size}."
                )
        return hidden_size

    def _get_tensor(self, model: Model, weight_name: str):
        TensorRepository.get_tensor(
            shards=model.shards,
            tensor_key=weight_name,
            device=self.env.device
        )

    def _get_all_model_tensors(
        self, 
        models_with_weights: Dict[Model, ModelWeight]
    ):
        models_with_tensors = {
        model: self._get_tensor(model, model_weight.name)
        for model, model_weight in models_with_weights.items()
        }

        return models_with_tensors
    
    def _get_base_model_tensor(
        self,
        base_model: Model,
        task_base_model_weight: ModelWeight
    ):
        return self._get_tensor(base_model, task_base_model_weight.name)
    
    def merge(
        self,
        base_model: Model,
        task_base_model_weight: ModelWeight,
        task_models_with_weights: Dict[Model, ModelWeight], 
        tokenizer: Tokenizer,
        method_config,
        sources
    ):
        base_model_tensor = self._get_base_model_tensor(
            base_model,
            task_base_model_weight
        )

        models_tensors = self._get_all_model_tensors(
            models_with_weights=task_models_with_weights,
        )

        all_tensors = {
            base_model: base_model_tensor,
            **models_tensors
        }

        hidden_dim = self._validate_tensor_shapes(
            base_model_weight=task_base_model_weight,
            tensors=all_tensors,
            base_model_layer_type=task_base_model_weight.layer_type
        )
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
        else:
            return method_config.method.merge(
                weight=task_base_model_weight,
                base_model_tensor=base_model_tensor,
                models_tensors=models_tensors,
                merge_method_settings=method_config.settings,
                base_model=base_model
            )