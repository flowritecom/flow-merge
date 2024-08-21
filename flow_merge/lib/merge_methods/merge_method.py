from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel

from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.model import Model


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
