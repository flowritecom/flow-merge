from typing import Dict, List, Tuple
import torch


def merge_linear(
        merge_method_settings: Dict[str, bool],
        tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool, str, str]],
) -> torch.Tensor:
    base: Tuple[torch.Tensor, float, bool] = [t for t in tensors_weights_pairs if t[2] is True][
        0]  # little bit dirty with the tuple for now
    base_tensor_dtype = base[0].dtype

    weights = [p[1] for p in tensors_weights_pairs if
               p[2] is False]  # assuming that base model must be last on the list?
    tensors = [p[0] for p in tensors_weights_pairs if p[2] is False]

    weights.append(base[1])
    tensors.append(base[0])

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
        if merge_method_settings["normalize"]:
            # relative weights
            merged_tensor = merged_tensor / weights_tensors.sum(dim=0)

    return merged_tensor.to(dtype=base_tensor_dtype)
