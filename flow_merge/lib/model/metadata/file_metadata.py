from pydantic import BaseModel
from typing import Optional
from huggingface_hub.hf_api import BlobLfsInfo

class FileMetadata(BaseModel):
    filename: str
    sha: Optional[str] = None
    blob_id: Optional[str] = None
    size: Optional[int] = None
    lfs: Optional[BlobLfsInfo] = None

