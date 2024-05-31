from typing import Optional, Dict, Any
from pydantic import BaseModel, field_validator

from ..hash import create_content_hash


class SnapshotHost(BaseModel):
    os: str
    system_architecture: str

class SnapshotMetadata(BaseModel):
    created_at: str
    library_version: str
    host: SnapshotHost
    sha: Optional[str]

    @field_validator('sha', mode='after')
    def compute_sha(cls, values: Dict[str, Any]):
        # Convert all fields except 'sha' to a dictionary
        data_dict = {k: v for k, v in values.items() if k != 'sha'}
        content_hash = create_content_hash(data_dict)
        values['sha'] = content_hash
        return values