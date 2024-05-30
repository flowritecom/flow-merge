from typing import Dict, Optional

import torch

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.merge_config import MergeConfig, Model
from flow_merge.lib.merge_methods.slerp import SlerpSettings
from flow_merge.lib.tensor_loader import TensorLoader


class Merger:
    def __init__(
        self,
        merge_config: MergeConfig,
        tensor_loaders: Dict[Model, TensorLoader],
        input_ids_mappings: Optional[Dict[Model, Dict[int, int]]],
    ) -> None:
        self.merge_config = merge_config
        self.tensor_loaders = tensor_loaders
        self.input_ids_mappings = input_ids_mappings
        self.merge_method = self.merge_config.method
        self.merge_method_settings = self.merge_config.method_config

    def _validate_tensor_shapes(
        self, weight: ModelWeight, tensors: Dict[Model, torch.Tensor]
    ) -> None:
        base_model_shape = tensors[self.merge_config.base_model].shape
        for model, tensor in tensors.items():
            if tensor.shape != base_model_shape:
                raise RuntimeError(
                    f"Tensor shape mismatch in '{weight.name}'. Expected {base_model_shape}, but {model.path} has a shape of {tensor.shape}."
                )

    def _get_model_tensors(self, weight: ModelWeight) -> Dict[Model, torch.Tensor]:
        tensors = {
            self.merge_config.base_model: self.tensor_loaders[
                self.merge_config.base_model
            ].get_tensor(weight.name)
        }
        for model in self.merge_config.models:
            tensors[model] = self.tensor_loaders[model].get_tensor(weight.name)
        return tensors

    def _map_and_merge_tensors(
        self, weight: ModelWeight, tensors: Dict[Model, torch.Tensor]
    ) -> torch.Tensor:
        base_model_dtype = tensors[self.merge_config.base_model].dtype
        all_models = self.merge_config.models + [self.merge_config.base_model]

        mapped_tensors = []
        mask_list = []
        weights_list = []

        for model in all_models:
            tensor = tensors[model]
            input_ids_map = self.input_ids_mappings[model]
            mapped_tensor = torch.zeros(
                (len(input_ids_map), tensor.shape[-1]), dtype=tensor.dtype
            )
            mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)

            for out_id, in_id in input_ids_map.items():
                if in_id >= 0:
                    mapped_tensor[out_id, :] = tensor[in_id, :]
                    mask[out_id] = True

            mapped_tensors.append(mapped_tensor)
            mask_list.append(mask)
            weights_list.append(
                1.0
                if isinstance(self.merge_method_settings, SlerpSettings)
                else self.merge_method_settings.weights[model]
            )

        stacked_mapped_tensors = torch.stack(mapped_tensors, dim=0)
        stacked_mask_tensor = torch.stack(mask_list, dim=0).unsqueeze(-1)
        weights_tensor = (
            torch.tensor(weights_list, dtype=stacked_mapped_tensors.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        total_weight = (stacked_mask_tensor * weights_tensor).sum(dim=0)
        scale = torch.where(
            total_weight.abs() < 1e-8, torch.zeros_like(total_weight), 1 / total_weight
        )

        merged_mapped_tensor = (
            stacked_mapped_tensors * weights_tensor * stacked_mask_tensor
        ).sum(dim=0) * scale
        return merged_mapped_tensor.to(dtype=base_model_dtype)

    def interpolate(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        tensors = self._get_model_tensors(weight)
        return self._map_and_merge_tensors(weight, tensors)

    def merge_weights(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        tensors = self._get_model_tensors(weight)
        self._validate_tensor_shapes(weight, tensors)

        merged_tensor = self.merge_method.merge(
            weight=weight,
            base_model_tensor=tensors[self.merge_config.base_model],
            models_tensors={
                model: tensors[model] for model in self.merge_config.models
            },
            merge_method_settings=self.merge_method_settings,
            base_model=self.merge_config.base_model,
        )

        return merged_tensor
