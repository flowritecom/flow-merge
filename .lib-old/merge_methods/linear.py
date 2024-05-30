from typing import Any, Dict, List, Optional, Type

import torch

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.constants import ConfigKeys, MergeMethodIdentifier
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_methods.merge_method import (
    BaseMergeMethodSettings,
    MergeMethod,
)
from flow_merge.lib.model import Model

logger = get_logger(__name__)


class Linear(MergeMethod):
    def merge(
        self,
        weight: ModelWeight,
        base_model_tensor: torch.Tensor,
        models_tensors: Dict[Model, torch.Tensor],
        merge_method_settings: BaseMergeMethodSettings,
        base_model: Model,
    ) -> torch.Tensor:
        base_tensor_dtype = base_model_tensor.dtype

        tensors: List[torch.Tensor] = []
        weights: List[float] = []
        for model, tensor in models_tensors.items():
            merge_weight = merge_method_settings.weights[model]
            weights.append(merge_weight)
            tensors.append(tensor)

        base_model_weight = merge_method_settings.weights[base_model]
        weights.append(base_model_weight)
        tensors.append(base_model_tensor)

        if set(weights) == {1.0}:
            # uniform soup
            merged_tensor = torch.stack(tensors, dim=0).sum(dim=0) / len(tensors)
        else:
            # weight average soup. If all weights are the same, this is equivalent to a simple average
            stacked_tensors = torch.stack(tensors, dim=0)
            weights_tensors = torch.tensor(
                weights, dtype=base_tensor_dtype, device=stacked_tensors.device
            )
            while len(weights_tensors.shape) < len(stacked_tensors.shape):
                weights_tensors.unsqueeze_(-1)

            merged_tensor = (stacked_tensors * weights_tensors).sum(dim=0)
            if merge_method_settings.normalize:
                # relative weights
                merged_tensor = merged_tensor / weights_tensors.sum(dim=0)

        return merged_tensor.to(dtype=base_tensor_dtype)
