import torch

from typing import Dict, List, Tuple, Any

from flow_merge.lib.model import Model
from flow_merge.lib.snapshot.data_architecture._normalized_slices import MergeMethodIdentifier


class InterpolationRunner:

    @staticmethod
    def _map_tensors(
            tensors: List[Tuple[torch.Tensor, float, bool]],
            input_ids_mappings: Dict[Model, Dict[int, int]],
            hidden_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped_tensors = []
        masks = []
        for (tensor, _, _) in tensors:
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
            tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool]],
            merge_method_name
    ):
        weights = [
            source[1]
            if source[1] or not merge_method_name == MergeMethodIdentifier.SLERP else 1.0
            for source in tensors_weights_pairs
        ]

        return torch.tensor(weights, dtype=torch.float32)

    @classmethod
    def interpolate(
            cls,
            all_tensors: List[Tuple[torch.Tensor, float, bool]],
            merge_method_name: str,
            input_ids_mappings: Dict[str, Dict[int, int]],
            hidden_dim: int
    ):
        base_tensor = [p for p in all_tensors if p[2] is True][0]
        mapped_tensors, masks = cls._map_tensors(
            all_tensors,
            input_ids_mappings,
            hidden_dim
        )

        weights = (
            cls._compute_weights(
                all_tensors,
                merge_method_name
            ).unsqueeze(-1).unsqueeze(-1)
        )

        total_weight = (masks.unsqueeze(-1) * weights).sum(dim=0)

        # FIXME confirm correctness of this
        scale = torch.where(total_weight.abs() < 1e-8, torch.tensor(0.0), 1 / total_weight)

        merged_tensor = (mapped_tensors * weights * masks.unsqueeze(-1)).sum(dim=0) * scale
        return merged_tensor.to(dtype=base_tensor[0].dtype)
