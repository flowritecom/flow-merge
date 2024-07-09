from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ValidationError, field_validator

from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.logger import get_logger
from flow_merge.lib.model import Model

logger = get_logger(__name__)


class BaseMergeMethodSettings(BaseModel):
    normalize: Optional[bool] = True

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
