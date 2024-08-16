import logging
from typing import Dict, Optional, List, Tuple

import torch
from pydantic import BaseModel, field_validator

from flow_merge.lib.model.architecture import ModelWeight
# from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_methods.merge_method import MergeMethod
from flow_merge.lib.model import Model


# FIXME new flow-merge repo format
# logger = get_logger(__name__)


class SlerpSettings(BaseModel):
    t: Optional[float] = 0.5
    weights: Optional[Dict[Model, float]] = {}

    @field_validator("weights", check_fields=False)
    @classmethod
    def validate_weights(cls, v):
        if v:
            for model, weight in v.items():
                if weight <= 0.0:
                    raise ValueError(
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}' from models if you don't want to use the model in the merge."
                    )
            return v

    @field_validator("t")
    @classmethod
    def validate_t(cls, v):
        if v:
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    "The interpolation parameter for spherical linear interpolation of 2 tensors `t` must be a value between 0.0 and 1.0"
                )
            return v


def merge_slerp(
        tensors_weights_pairs: List[Tuple[torch.Tensor, float, bool]],
        merge_method_settings: SlerpSettings,
) -> torch.Tensor:
    # little bit dirty with the tuple for now
    base: Tuple[torch.Tensor, float, bool] = [t for t in tensors_weights_pairs if t[2] is True][0]
    base_tensor_dtype = base[0].dtype

    v0 = base[0]
    # Only 1 model is supported for slerp fixme: let user know that other sources are being ignored
    v1 = tensors_weights_pairs[0][0]

    t = merge_method_settings.t
    DOT_THRESHOLD = 0.9995
    eps: float = 1e-8

    v0_copy = v0.clone()
    v1_copy = v1.clone()
    # Normalize the vectors to get the directions and angles
    v0 = _normalize(v0, eps)
    v1 = _normalize(v1, eps)
    # Dot product with the normalized vectors
    dot = torch.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if torch.abs(dot) >= torch.tensor(DOT_THRESHOLD, dtype=dot.dtype):
        logging.info(
            f"Vectors v0={v0.__hash__()} & v1={v1.__hash__()} are colineal, using lerp instead of slerp."
        )
        return torch.lerp(v0_copy, v1_copy, t)
    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2.to(dtype=base_tensor_dtype)


def _normalize(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return tensor / torch.norm(tensor) if torch.norm(tensor) > eps else tensor
