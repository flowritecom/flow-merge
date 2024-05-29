from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel

from flow_merge.lib.architecture import ModelWeight, get_all_weights
from flow_merge.lib.merge_config import MergeConfig, Model
from flow_merge.lib.merge_methods.slerp import SlerpSettings
from flow_merge.lib.tensor_loader import TensorRepository
from flow_merge.lib.tensor_writer import TensorWriter


def validate_tensor_shapes(
    weight: ModelWeight, tensors: Dict[Model, torch.Tensor], layer_type: str
):
    """
    Validates that all tensors have the same shape based on the layer type.

    Args:
        weight (ModelWeight): The weight configuration.
        tensors (Dict[Model, torch.Tensor]): Tensors to validate.
        layer_type (str): The type of layer ("embedding" or other).

    Returns:
        int: The hidden size of the tensors.

    Raises:
        RuntimeError: If tensor shapes do not match.
    """
    hidden_size = next(iter(tensors.values())).shape[
        1 if layer_type == "embedding" else 0
    ]
    for model, tensor in tensors.items():
        current_size = tensor.shape[1] if layer_type == "embedding" else tensor.shape[0]
        if current_size != hidden_size:
            raise RuntimeError(
                f"Tensor shape mismatch in '{weight.name}'. Expected {hidden_size}, but {model.path} has {current_size}."
            )
    return hidden_size


def map_tensors(
    tensors: Dict[Model, torch.Tensor],
    input_ids_mappings: Dict[Model, Dict[int, int]],
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Maps tensors according to input ID mappings.

    Args:
        tensors (Dict[Model, torch.Tensor]): Original tensors.
        input_ids_mappings (Dict[Model, Dict[int, int]]): Input ID mappings.
        hidden_dim (int): Hidden dimension size.

    Returns:
        (torch.Tensor, torch.Tensor): Mapped tensors and masks.
    """
    mapped_tensors = []
    masks = []
    for model, tensor in tensors.items():
        input_ids_map = input_ids_mappings[model]
        mapped_tensor = torch.zeros(
            (len(input_ids_map), hidden_dim), dtype=tensor.dtype
        )
        mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)

        for out_id, in_id in input_ids_map.items():
            if in_id >= 0:
                mapped_tensor[out_id] = tensor[in_id]
                mask[out_id] = True

        mapped_tensors.append(mapped_tensor)
        masks.append(mask)

    return torch.stack(mapped_tensors), torch.stack(masks)

# WARNING: these are the biases, not model weights
def compute_weights(
    method_config, models: List[Model], base_model: Model
) -> torch.Tensor:
    """
    Computes weights for models.

    Args:
        method_config: Configuration for the merge method.
        models ([Model]): List of models.
        base_model (Model): The base model.

    Returns:
        torch.Tensor: Computed weights.
    """
    weights = [
        method_config.weights[model]
        if not isinstance(method_config, SlerpSettings)
        else 1.0
        for model in models + [base_model]
    ]
    return torch.tensor(weights, dtype=torch.float32)


# WARNING -> GPT4o's understanding of what we did before -> Fix ?
def interpolate_tensors(
    tensor_loaders: Dict[Model, TensorRepository],
    input_ids_mappings: Optional[Dict[Model, Dict[int, int]]],
    method_config,
    weight: ModelWeight,
    base_model: Model,
    models: List[Model],
) -> torch.Tensor:
    """
    Interpolates tensors based on input ID mappings.

    Args:
        tensor_loaders (Dict[Model, TensorRepository]): Dictionary of tensor loaders.
        input_ids_mappings (Optional[Dict[Model, Dict[int, int]]]): Dictionary of input ID mappings.
        method_config: Configuration for the merge method.
        weight (ModelWeight): The weight configuration.
        base_model (Model): The base model.
        models ([Model]): List of models.

    Returns:
        torch.Tensor: Interpolated tensor.
    """
    all_models = models + [base_model]
    all_tensors = {
        model: tensor_loaders[model].get_tensor(weight.name) for model in all_models
    }
    hidden_dim = validate_tensor_shapes(weight, all_tensors, weight.layer_type)

    mapped_tensors, masks = map_tensors(all_tensors, input_ids_mappings, hidden_dim)
    weights = (
        compute_weights(method_config, models, base_model).unsqueeze(-1).unsqueeze(-1)
    )

    total_weight = (masks.unsqueeze(-1) * weights).sum(dim=0)
    # FIXME: this is one that needs to be tested well, we didn't use torch.where before
    scale = torch.where(total_weight.abs() < 1e-8, torch.tensor(0.0), 1 / total_weight)

    merged_tensor = (mapped_tensors * weights * masks.unsqueeze(-1)).sum(dim=0) * scale
    return merged_tensor.to(dtype=all_tensors[base_model].dtype)


class Merge(BaseModel):
    """
    The Merge class is responsible for merging weights based on a specified merge configuration.

    Attributes:
        merge_config: The merge configuration.
    """

    merge_config: MergeConfig

    class Config:
        frozen = True

    def merge_weights(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        """
        Merges the weights of multiple models based on the specified merge configuration and method.

        Args:
            weight (ModelWeight): The weight configuration.

        Returns:
            Dict[str, torch.Tensor]: Merged tensor.
        """
        base_model_tensor = self.merge_config.tensor_loaders[
            self.merge_config.base_model
        ].get_tensor(weight.name)
        models_tensors = {
            model: self.merge_config.tensor_loaders[model].get_tensor(weight.name)
            for model in self.merge_config.models
        }

        validate_tensor_shapes(weight, models_tensors, weight.layer_type)

        merged_tensor = self.merge_config.method.merge(
            weight=weight,
            base_model_tensor=base_model_tensor,
            models_tensors=models_tensors,
            merge_method_settings=self.merge_config.method_config,
            base_model=self.merge_config.base_model,
        )

        return merged_tensor

    def interpolate(self, weight: ModelWeight) -> Dict[str, torch.Tensor]:
        """
        Interpolates the weights of embedding and lm head layers when there are input ID mappings.

        Args:
            weight (ModelWeight): The weight configuration.

        Returns:
            Dict[str, torch.Tensor]: Interpolated tensor.
        """
        return interpolate_tensors(
            tensor_loaders=self.merge_config.tensor_loaders,
            input_ids_mappings=self.merge_config.tokenizer.input_ids_mappings,
            method_config=self.merge_config.method_config,
            weight=weight,
            base_model=self.merge_config.base_model,
            models=self.merge_config.models,
        )

    def process_and_save_weights(self) -> None:
        """
        This function processes and saves model weights using the TensorWriter.

        Args:
            model_arch: The architecture of the model.
            merge_config: Configuration for merging tensors.
            tokenizer: The tokenizer used for processing input ids mappings.
            merger: An object that handles the merging and interpolation of weights.

        Returns:
            None
        """
        # Initialize writer
        # FIXME: the arguments are wrong!
        with TensorWriter(merge_config=self.merge_config) as writer:
            for weight in self.merge_config.base_architecture.get_all_weights():
                if self.merge_config.tokenizer.input_ids_mappings and (
                    weight.layer_type == "embedding" or weight.layer_type == "head"
                ):
                    merged_tensor = self.interpolate(weight)
                else:
                    merged_tensor = self.merge_weights(weight)
                writer.save_tensor(weight=weight, tensor=merged_tensor, clone=False)
            writer.finish()
