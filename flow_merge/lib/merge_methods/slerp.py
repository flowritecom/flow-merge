from typing import Dict, Optional

import torch
from pydantic import BaseModel, field_validator

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_methods.merge_method import MergeMethod
from flow_merge.lib.model import Model

logger = get_logger(__name__)


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


class Slerp(MergeMethod):
    """
    This class implements the Slerp algorithm for merging model weights. It supports merging
    two model tensors using a specified interpolation parameter (t).

    The Slerp algorithm is used when the vectors are not colineal (dot product < DOT_THRESHOLD).
    If the vectors are colineal, the class falls back to using linear interpolation (lerp) instead.
    """

    def merge(
        self,
        weight: ModelWeight,
        base_model_tensor: torch.Tensor,
        models_tensors: Dict[Model, torch.Tensor],
        merge_method_settings: SlerpSettings,
        base_model: Model,
    ) -> torch.Tensor:
        base_tensor_dtype = base_model_tensor.dtype

        v0 = base_model_tensor
        v1 = list(models_tensors.values())[0]  # Only 1 model is supported for slerp

        merged_tensor = self._slerp(
            weight=weight, t=merge_method_settings.t, v0=v0, v1=v1
        )

        return merged_tensor.to(dtype=base_tensor_dtype)

    def _normalize(self, tensor: torch.Tensor, eps: float) -> torch.Tensor:
        return tensor / torch.norm(tensor) if torch.norm(tensor) > eps else tensor

    def _slerp(
        self,
        weight: ModelWeight,
        t: float,
        v0: torch.Tensor,
        v1: torch.Tensor,
        DOT_THRESHOLD=0.9995,
        eps: float = 1e-8,
    ):
        """
        Spherical linear interpolation from https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c#file-pytorch-tensor-slerp-py

        Args:
            weight: Model weight. e.g. 'embed_tokens.weight'
            t: Float value between 0.0 and 1.0.
            v0: Starting vector.
            v1: Final vector.
            DOT_THRESHOLD: Threshold for considering the two vectors as
                            colineal. Not recommended to alter this.
            eps: Small value to avoid division by zero during normalization.

        Returns:
            Interpolation vector between v0 and v1.
        """
        # Copy the vectors to reuse them later
        v0_copy = v0.clone()
        v1_copy = v1.clone()
        # Normalize the vectors to get the directions and angles
        v0 = self._normalize(v0, eps)
        v1 = self._normalize(v1, eps)
        # Dot product with the normalized vectors
        dot = torch.sum(v0 * v1)
        # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
        if torch.abs(dot) >= torch.tensor(DOT_THRESHOLD, dtype=dot.dtype):
            logger.info(
                f"Vectors are colineal, using lerp instead of slerp for {weight.name}."
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
        return v2
