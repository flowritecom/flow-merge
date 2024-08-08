from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ValidationError, field_validator

# from flow_merge.lib.logger import get_logger
from flow_merge.lib.model import Model

# FIXME new flow-merge repo format
# logger = get_logger(__name__)


class BaseMergeMethodSettings(BaseModel):
    normalize: Optional[bool] = True

class MergeMethod(ABC):
    @abstractmethod
    def merge(self, slice) -> torch.Tensor:
        pass
