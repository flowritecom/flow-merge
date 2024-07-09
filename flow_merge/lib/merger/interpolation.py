import torch

from typing import Dict, List, Tuple

from flow_merge.lib.model import Model
from flow_merge.lib.merge_methods.slerp import SlerpSettings
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource

class InterpolationRunner:

    @staticmethod
    def _map_tensors(
        cls,
        tensors: Dict[Model, torch.Tensor],
        input_ids_mappings: Dict[Model, Dict[int, int]],
        hidden_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped_tensors = []
        masks = []
        for model, tensor in tensors.items():
            input_ids_map = input_ids_mappings[model]
            mapped_tensor = torch.zeros(
                (len(input_ids_map), hidden_size), dtype=tensor.dtype
            )
            mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)

            for out_id, in_id in input_ids_map.items():
                if in_id >= 0:
                    mapped_tensor[out_id] = tensor[in_id]
                    mask[out_id] = True

            mapped_tensors.append(mapped_tensor)
            masks.append(mask)

        return torch.stack(mapped_tensors), torch.stack(masks)
    
    @staticmethod
    def _compute_weights(
        cls,
        sources: List[NormalizedSource],
        method_config
    ):
        weights = [
            source.weight
            if source.weight or not isinstance(method_config.settings, SlerpSettings) else 1.0
            for source in sources
        ]

        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def interpolate(
        cls,
        base_model: Model,
        all_tensors: Dict[Model, torch.Tensor],
        method_config,
        input_ids_mappings: Dict[Model, Dict[int, int]],
        sources: List[NormalizedSource],
        hidden_dim: int
    ):
        mapped_tensors, masks = cls._map_tensors(
            all_tensors,
            input_ids_mappings,
            hidden_dim
        )

        weights = (
            cls._compute_weights(
                sources,
                method_config
            ).unsqueeze(-1).unsqueeze(-1)
        )

        total_weight = (masks.unsqueeze(-1) * weights).sum(dim=0)

        # FIXME confirm correctness of this
        scale = torch.where(total_weight.abs() < 1e-8, torch.tensor(0.0), 1 / total_weight)

        merged_tensor = (mapped_tensors * weights * masks.unsqueeze(-1)).sum(dim=0) * scale
        return merged_tensor.to(dtype=all_tensors[base_model].dtype)
