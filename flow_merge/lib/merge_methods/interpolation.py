import torch
from typing import List, Tuple, Dict
from flow_merge.lib.merge_methods import MergeMethodIdentifier


def validate_tensor_shape(
        all_tensors: List[Tuple[torch.Tensor, float, bool, str, str]],
        dim_index: int, 
        layer_name: str
):
    hidden_size = None
    for tensor, _, _, _, _ in all_tensors:
        hidden_size = tensor.shape[dim_index] if hidden_size is None else hidden_size

        if tensor.shape[dim_index] != hidden_size:
            raise RuntimeError(
                    f"Tensor shape mismatch in '{layer_name}'. Expected {hidden_size}, but got a hidden size of {tensor.shape[dim_index]}."
                )


def map_tensors(
        all_tensors: List[Tuple[torch.Tensor, float, bool, str, str]],
        input_ids_mappings: Dict[str, Dict[int, int]],
):
    mapped_tensors = []
    mask_list = []
    for tensor, _, _, model, _ in all_tensors:
        input_ids_map = input_ids_mappings[model]
        mapped_tensor = torch.zeros(
                (len(input_ids_map), tensor.shape[-1]), dtype=tensor.dtype
            )
        mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)
        for out_id in input_ids_map:
            in_id = input_ids_map[out_id]

            if in_id < 0 or len(tensor.shape) < 2:
                    continue

            mapped_tensor[out_id, :] = tensor[in_id, :]
            mask[out_id] = True

        mapped_tensors.append(mapped_tensor)
        mask_list.append(mask)

    return mapped_tensors, mask_list


def compute_weights(
        all_tensors: List[Tuple[torch.Tensor, float, bool, str, str]],
        merge_method_name: str
):
    weights = [
            source[1]
            if source[1] and not merge_method_name == MergeMethodIdentifier.SLERP else 1.0
            for source in all_tensors
        ]
    
    return weights


def interpolate(
        all_tensors: List[Tuple[torch.Tensor, float, bool, str, str]],
        input_ids_mappings: Dict[str, Dict[int, int]],
        merge_method_name: str
)-> torch.Tensor:
    base_tensor = [p for p in all_tensors if p[2] is True][0]
    base_tensor_dtype = base_tensor[0].dtype
    layer_name = base_tensor[4] #source.layer

    validate_tensor_shape(
        all_tensors=all_tensors,
        dim_index=1 if "embed" in layer_name else 0,
        layer_name=layer_name
    )
    mapped_tensors, mask_list = map_tensors(all_tensors, input_ids_mappings)
    weights = compute_weights(all_tensors, merge_method_name)

    stacked_mapped_tensors = torch.stack(mapped_tensors, dim=0)
    stacked_mask_tensor = torch.stack(mask_list, dim=0).unsqueeze(-1)
    weights_tensor = (
        torch.tensor(weights, dtype=stacked_mapped_tensors.dtype)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    total_weight = (stacked_mask_tensor * weights_tensor).sum(dim=0)
    scale = 1 / total_weight
    scale[total_weight.abs() < 1e-8] = 0

    merged_mapped_tensor = (
        stacked_mapped_tensors * weights_tensor * stacked_mask_tensor
    ).sum(dim=0) * scale

    return merged_mapped_tensor.to(dtype=base_tensor_dtype)

