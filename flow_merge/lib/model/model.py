from pathlib import Path
from typing import List, NewType, Optional, Dict
from pydantic import BaseModel
from flow_merge.lib.model.architecture import ModelArchitecture
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.tensor.loader import ShardFile

ModelId = NewType("ModelId", str)


class Model(BaseModel, arbitrary_types_allowed=True):
    id: ModelId
    path: Path
    metadata: ModelMetadata
    file_to_tensor_index: Optional[Dict]
    shards: List[ShardFile]
    architecture: ModelArchitecture

    is_partial: bool = False

    def __hash__(self):
        return hash((self.id, self.metadata.sha))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.id == other.id and self.metadata.sha == other.metadata.sha
        return False

    def __str__(self):
        if self.metadata.sha:
            return f"{self.id}@{self.metadata.sha}"
        return str(self.id)
