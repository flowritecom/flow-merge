from enum import Enum
from typing import Any, Dict, Optional, Type, Literal

from pydantic import BaseModel, Field, model_validator
from flow_merge.lib.merge_methods import MergeMethodIdentifier


class MethodSettings(BaseModel, arbitrary_types_allowed=True):
    merge_method: MergeMethodIdentifier = Field(alias="method")
    scaling_coefficient: Optional[float] = None
    normalize: Optional[bool] = None
    p: Optional[float] = None
    t: Optional[float] = None
    top_k: Optional[float] = None
    weights: Optional[Dict[Any, float]] = {}

    def _unpack(self):
        return (self.merge_method, self.scaling_coefficient, self.normalize,
                self.p,
                self.t,
                self.top_k,
                self.weights,)

    @model_validator(mode="after")
    def validate_method_and_params(self):
        if params:
            self._validate_weights(self.weights)
            self._validate_scaling_coefficient(self.scaling_coefficient)
            self._validate_p(self.p)
            self._validate_t(self.t)
            self._validate_top_k(self.top_k)
        return self

    def _validate_weights(self, weights: Optional[Dict[Any, float]]):
        if weights:
            total_weight = sum(weights.values())
            if total_weight > 1.0:
                raise ValueError("The combined weights cannot exceed 1.0.")
            for model, weight in weights.items():
                if weight <= 0.0:
                    raise ValueError(
                        f"Weight for model '{model}' must be greater than 0."
                    )

    def _validate_scaling_coefficient(self, scaling_coefficient: Optional[float]):
        if self.merge_method == MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC:
            if (
                    scaling_coefficient is not None
                    and not 0.0 <= scaling_coefficient <= 1.0
            ):
                raise ValueError(
                    "Scaling coefficient should be a value between 0.0 and 1.0."
                )

    def _validate_p(self, p: Optional[float]):
        if self.merge_method == MergeMethodIdentifier.DARE_TIES_MERGING:
            if p is not None and not 0.0 <= p <= 1.0:
                raise ValueError("p should be between 0.0 and 1.0.")

    def _validate_t(self, t: Optional[float]):
        if self.merge_method == MergeMethodIdentifier.SLERP:
            if t is not None and not 0.0 <= t <= 1.0:
                raise ValueError(
                    "The interpolation parameter t must be a value between 0.0 and 1.0."
                )

    def _validate_top_k(self, top_k: Optional[float]):
        if self.merge_method == MergeMethodIdentifier.TIES_MERGING:
            if top_k is not None and not 0.0 <= top_k <= 1.0:
                raise ValueError("top_k should be a value between 0.0 and 1.0.")
