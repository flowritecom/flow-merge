from pydantic import BaseModel

from ._metadata import SnapshotMetadata
from ._settings import MergeSettings
from ._normalized_slices import NormalizedSlices

class Snapshot(BaseModel):
    sha: str
    metadata: SnapshotMetadata
    settings: MergeSettings
    normalized: NormalizedSlices

