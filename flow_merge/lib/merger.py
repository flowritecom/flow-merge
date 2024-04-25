from typing import Dict, Optional

import torch

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.merge_config import MergeConfig, Model
from flow_merge.lib.merge_methods.slerp import SlerpSettings
from flow_merge.lib.tensor_loader import TensorLoader


class Merger:
    """
    The Merger class is responsible for merging weights based on a specified merge configuration.

    Args:
        merge_config: An instance of the `MergeConfig` class, which holds the configuration for the merging process.
        tensor_loaders: A dictionary of `TensorLoader` instances, where the keys are the models and the values are the corresponding `TensorLoader` objects.
        input_ids_mappings: An optional dictionary that contains input ID mappings for each model from differences in the tokenizers.

    Attributes:
        merge_config: The merge configuration.
        tensor_loaders: The dictionary of `TensorLoader` instances.
        input_ids_mappings: The optional dictionary of input ID mappings.
        merge_method: The merge method specified in the `merge_config`.
        merge_method_settings: The configuration for the merge method specified in the `merge_config` that contains global parameters for the merge method.
    """

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

    def interpolate(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        """
        Interpolates the weights of embedding and lm head layers when there are inputs ids mappings.
        """
        base_model_dtype = (
            self.tensor_loaders[self.merge_config.base_model]
            .get_tensor(weight.name)
            .dtype
        )
        all_models = self.merge_config.models + [self.merge_config.base_model]

        mapped_tensors = []
        mask_list = []
        weights_list = []

        # ! Note that if Embedding's num_embeddings is larger than vocab size for distributed training reasons,
        # ! the embeddings and lm_head will be rectified.
        # ! e.g. Qwen1.5 models have a vocab size of 151646 but the num of embeddings are 151936.
        # ! The resulting tensor will be of shape (151646, hidden_size) for the embeddings and (hidden_size, 151646) for the lm_head.
        all_tensors = {}
        for model in all_models:
            tensor_loader = self.tensor_loaders[model]
            tensor = tensor_loader.get_tensor(weight.name)
            all_tensors[model] = tensor

        # Tensor shape validations
        if weight.layer_type == "embedding":
            # Validate that the second dimension of tensors are the same
            hidden_size = None
            for model, tensor in all_tensors.items():
                if hidden_size is None:
                    hidden_size = tensor.shape[1]
                else:
                    if tensor.shape[1] != hidden_size:
                        raise RuntimeError(
                            f"Tensor shape mismatch in '{weight.name}'. Expected {hidden_size}, but {model.path} has a hidden size of {tensor.shape[1]}."
                        )
        else:
            hidden_size = None
            for model, tensor in all_tensors.items():
                if hidden_size is None:
                    hidden_size = tensor.shape[0]
                else:
                    if tensor.shape[0] != hidden_size:
                        raise RuntimeError(
                            f"Tensor shape mismatch in '{weight.name}'. Expected {hidden_size}, but {model.path} has a hidden size of {tensor.shape[0]}."
                        )

        for model, tensor in all_tensors.items():
            input_ids_map = self.input_ids_mappings[model]
            mapped_tensor = torch.zeros(
                (len(input_ids_map), tensor.shape[-1]), dtype=tensor.dtype
            )
            mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)

            for out_id in input_ids_map:
                in_id = input_ids_map[out_id]
                if in_id < 0:
                    continue
                mapped_tensor[out_id, :] = tensor[in_id, :]
                mask[out_id] = True

            mapped_tensors.append(mapped_tensor)
            mask_list.append(mask)
            # SlerpSettings does not use BaseMergeMethodConfig - special case
            if not type(self.merge_method_settings) == SlerpSettings:
                weights_list.append(
                    self.merge_method_settings.weights[model]
                )  # * merge weight for the model - float
            else:
                weights_list.append(1.0)  # default - full model

        stacked_mapped_tensors = torch.stack(mapped_tensors, dim=0)
        stacked_mask_tensor = torch.stack(mask_list, dim=0).unsqueeze(-1)
        weights_tensor = (
            torch.tensor(weights_list, dtype=stacked_mapped_tensors.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        total_weight = (stacked_mask_tensor * weights_tensor).sum(dim=0)
        scale = 1 / total_weight
        scale[total_weight.abs() < 1e-8] = 0

        merged_mapped_tensor = (
            stacked_mapped_tensors * weights_tensor * stacked_mask_tensor
        ).sum(dim=0) * scale

        return merged_mapped_tensor.to(dtype=base_model_dtype)

    def merge_weights(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        """
        Merges the weights of multiple models based on the specified merge configuration and method.
        """
        base_model_tensor = self.tensor_loaders[
            self.merge_config.base_model
        ].get_tensor(weight.name)
        models_tensors = {}
        for model in self.merge_config.models:
            tensor_loader = self.tensor_loaders[model]
            models_tensors[model] = tensor_loader.get_tensor(weight.name)

        # Validate tensor shapes
        base_model_shape = base_model_tensor.shape
        for model, tensor in models_tensors.items():
            if tensor.shape != base_model_shape:
                raise RuntimeError(
                    f"Tensor shape mismatch in '{weight.name}'. Expected {base_model_shape}, but {model.path} has a shape of {tensor.shape}."
                )

        merged_tensor = self.merge_method.merge(
            weight=weight,
            base_model_tensor=base_model_tensor,
            models_tensors=models_tensors,
            merge_method_settings=self.merge_method_settings,
            base_model=self.merge_config.base_model,
        )

        return merged_tensor
