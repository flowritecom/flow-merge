from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from flow_merge.lib.constants import MergeMethodIdentifier

class MergeMethodSettings(BaseModel):
    merge_method: MergeMethodIdentifier = Field(alias="method")
    scaling_coefficient: Optional[float] = None
    normalize: Optional[bool] = None
    p: Optional[float] = None
    t: Optional[float] = None
    top_k: Optional[float] = None
    weights: Optional[Dict[Any, float]] = {}

    @classmethod
    @field_validator("merge_method")
    def validate_merge_method(cls, v):
        if v == MergeMethodIdentifier.PASSTHROUGH.value:
            raise ValidationError("Passthrough merging method is not implemented yet")
        # FIXME: by bringing method_classes to constants
        # if not v in method_classes:
        if False:
            valid_methods = [method.value for method in MergeMethodIdentifier]
            raise ValidationError(
                f"Invalid method: '{v}'. Valid methods are: {', '.join(valid_methods)}"
            )
        return v

    @classmethod
    @model_validator(mode='before')
    def validate_parameters(cls, values):
        merge_method = values.get("method")

        if merge_method == "task_arithmetic":
            scaling_coefficient = values.get("scaling_coefficient")
            if scaling_coefficient is not None and not 0.0 <= scaling_coefficient <= 1.0:
                raise ValidationError(
                    "scaling_coefficient",
                    "Scaling coefficient should be a value between 0.0 and 1.0. It is used to scale the task vectors before adding to the base model tensor. It is referred to as scaling term in the paper Editing Models with Task Arithmetic (https://arxiv.org/abs/2212.04089)",
                )

        if merge_method in ["task_arithmetic", "ties_merging", "dare_ties_merging"]:
            weights = values.get("weights", {})
            for model, weight in weights.items():
                if weight <= 0.0:
                    raise ValidationError(
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}' from models if you don't want to use the model in the merge."
                    )

        if merge_method == "dare_ties_merging":
            p = values.get("p")
            if p is not None and not 0.0 <= p <= 1.0:
                raise ValidationError(
                    "p",
                    "p should be between 0.0 and 1.0. It represents the drop rate for random binary mask for each task vector as described in the DARE approach in the paper Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (https://arxiv.org/abs/2311.03099).",
                )

        if merge_method == "ties_merging":
            top_k = values.get("top_k")
            if top_k is not None and not 0.0 <= top_k <= 1.0:
                raise ValidationError(
                    "top_k",
                    "top_k should be a value between 0.0 and 1.0. It represents the fraction of top values to keep in the task vectors based on their magnitude as described in the paper Resolving Interference When Merging Models (https://arxiv.org/abs/2306.01708)."
                )

        if merge_method == "slerp":
            t = values.get("t")
            if t is not None and not 0.0 <= t <= 1.0:
                raise ValidationError(
                    "The interpolation parameter for spherical linear interpolation of 2 tensors `t` must be a value between 0.0 and 1.0"
                )

        return values
