from typing import List, Tuple

from torch import Tensor


def merge_passthrough(tensors_weights_pairs: List[Tuple[Tensor, float, bool]], merge_method_settings: dict):
    return tensors_weights_pairs.pop()[0]
