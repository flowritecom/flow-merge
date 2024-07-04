from typing import Optional, Dict, Any
from pydantic import BaseModel, model_validator

from ..hash import create_content_hash


class SnapshotHost(BaseModel):
    os: str
    system_architecture: str

class SnapshotMetadata(BaseModel):
    created_at: str
    library_version: str
    host: SnapshotHost
    sha: Optional[str] = None

    @model_validator(mode="after")
    def compute_sha(self):
        # Convert all fields except 'sha' to a dictionary
        data_dict = self.model_dump()
        data_dict.pop("sha")
        content_hash = create_content_hash(data_dict)
        self.sha = content_hash

        return self
        