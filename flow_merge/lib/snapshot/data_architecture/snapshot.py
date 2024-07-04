from pydantic import BaseModel

from flow_merge.lib.snapshot.data_architecture._metadata import SnapshotMetadata
from flow_merge.lib.snapshot.data_architecture._settings import MergeSettings
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlices

class Snapshot(BaseModel):
    sha: str
    metadata: SnapshotMetadata
    settings: MergeSettings
    normalized: NormalizedSlices

