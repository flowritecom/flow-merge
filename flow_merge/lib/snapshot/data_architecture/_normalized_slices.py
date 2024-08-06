from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class MergeMethodIdentifier(str, Enum):
    ADDITION_TASK_ARITHMETIC = "addition-task-arithmetic"
    TIES_MERGING = "ties-merging"
    SLERP = "slerp"
    DARE_TIES_MERGING = "dare-ties-merging"
    MODEL_SOUP = "model-soup"
    PASSTHROUGH = "passthrough"
    INTERPOLATE = "interpolate"


class NormalizedSource(BaseModel):
    weight: Optional[float] = None
    model: Optional[str]
    layer: Optional[str]
    is_base: Optional[bool]


class MergeMethod(BaseModel):
    name: MergeMethodIdentifier
    params: Optional[Dict[str, Any]] = None


class NormalizedSlice(BaseModel):
    merge_method: MergeMethod
    sources: List[NormalizedSource]


class NormalizedSlices(BaseModel):
    slices: List[NormalizedSlice]
