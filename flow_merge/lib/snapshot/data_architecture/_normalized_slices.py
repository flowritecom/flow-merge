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
    weight: Optional[int] = None
    model: Optional[str]
    layer: Optional[str]
    # FIXME base_model is now is_base
    base_model: Optional[bool]

class NormalizedSlice(BaseModel):
    # FIXME merge method is a new type with {name, params}
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

    