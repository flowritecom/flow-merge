from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ValidationError, field_validator

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.logger import get_logger
from flow_merge.lib.model import Model

logger = get_logger(__name__)


class BaseMergeMethodSettings(BaseModel):
    normalize: Optional[bool] = True
    weights: Optional[Dict[Model, float]] = {}

    @field_validator("weights", check_fields=False)
    @classmethod
    def validate_weights(cls, v):
        if v:
            for model, weight in v.items():
                if weight <= 0.0:
                    raise ValidationError(
                        "weights",
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}' from models if you don't want to use the model in the merge.",
                    )
            return v


class MergeMethod(ABC):
    @abstractmethod
    def merge(
        self,
        weight: ModelWeight,
        base_model_tensor: torch.Tensor,
        models_tensors: Dict[Model, torch.Tensor],
        method_config: Any,
        base_model: Model,
    ) -> torch.Tensor:
        pass
