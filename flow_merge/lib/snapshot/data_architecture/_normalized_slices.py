from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, field_validator

from ..hash import create_content_hash

class MergeMethodIdentifier(str, Enum):
    ADDITION_TASK_ARITHMETIC = "addition-task-arithmetic"
    TIES_MERGING = "ties-merging"
    SLERP = "slerp"
    DARE_TIES_MERGING = "dare-ties-merging"
    MODEL_SOUP = "model-soup"
    PASSTHROUGH = "passthrough"

class NormalizedSource(BaseModel):
    model: str
    layer: str
    base_model: Optional[bool]

class NormalizedSlice(BaseModel):
    merge_method: MergeMethodIdentifier
    sources: List[NormalizedSource]

class NormalizedSlices(BaseModel):
    slices: List[NormalizedSlice]
    sha: Optional[str]

    @field_validator('sha', mode='after')
    def compute_sha(cls, values: Dict[str, Any]):
        # Convert all fields except 'sha' to a dictionary
        data_dict = {k: v for k, v in values.items() if k != 'sha'}
        content_hash = create_content_hash(data_dict)
        values['sha'] = content_hash
        return values

    