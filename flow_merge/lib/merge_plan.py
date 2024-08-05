import datetime
import os
import platform
from typing import List, Any
# Hello!
from pydantic import BaseModel

from flow_merge.lib.loaders.normalizer import Slice
from flow_merge.lib.merge_settings import MergeSettings


class MergePlan(BaseModel):
    created_at: datetime.datetime
    host: str
    merge_config: MergeSettings
    slices: List[Slice]
    lib_version: str
    sha: str

    def __init__(self, **data: Any):
        self.created_at = datetime.datetime.now()
        self.host = platform.node().__str__()
        super().__init__(**data)


