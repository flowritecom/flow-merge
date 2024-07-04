from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, model_validator

from ..hash import create_content_hash

class MergeMethodIdentifier(str, Enum):
    ADDITION_TASK_ARITHMETIC = "addition-task-arithmetic"
    TIES_MERGING = "ties-merging"
    SLERP = "slerp"
    DARE_TIES_MERGING = "dare-ties-merging"
    MODEL_SOUP = "model-soup"
    PASSTHROUGH = "passthrough"

class NormalizedSource(BaseModel):
    model: Optional[str]
    layer: Optional[str]
    base_model: Optional[bool]

class NormalizedSlice(BaseModel):
    merge_method: Optional[MergeMethodIdentifier]
    sources: List[NormalizedSource]

class NormalizedSlices(BaseModel):
    slices: List[NormalizedSlice]
    sha: Optional[str]

    @model_validator(mode="after")
    def compute_sha(self):
        # Convert all fields except 'sha' to a dictionary
        data_dict = self.model_dump()
        data_dict.pop("sha")
        content_hash = create_content_hash(data_dict)
        self.sha = content_hash

        return self

    