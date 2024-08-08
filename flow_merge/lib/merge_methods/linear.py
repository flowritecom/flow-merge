from typing import Any, Dict, List, Optional, Type

import torch

from flow_merge.lib.merge_methods.merge_method import (
    BaseMergeMethodSettings,
    MergeMethod,
)
from flow_merge.lib.model import Model


class Linear(MergeMethod):
    def merge(self, slice) -> torch.Tensor:
        base_model_tensor = [src.model.tensor for src in slice.sources if src.is_base]
        base_model_weight = [src.model.weight for src in slice.sources if src.is_base]
        settings = slice.merge_method.settings

        tensors = [src.tensor for src in slice.sources if not src.is_base] + base_model_tensor
        weights = [src.weight for src in slice.sources if not src.is_base] + base_model_weight

        base_tensor_dtype = base_model_tensor.dtype

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
            if settings.normalize:
                # relative weights
                merged_tensor = merged_tensor / weights_tensors.sum(dim=0)

        return merged_tensor.to(dtype=base_tensor_dtype)
