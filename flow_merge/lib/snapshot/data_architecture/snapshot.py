from typing import Optional
from pydantic import BaseModel

from flow_merge.lib.snapshot.data_architecture._metadata import SnapshotMetadata
from flow_merge.lib.snapshot.data_architecture._settings import MergeSettings
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlices

class Snapshot(BaseModel):
    sha: str
    metadata: SnapshotMetadata
    settings: MergeSettings
    normalized: NormalizedSlices
    # FIXME remove optional once finalized in normalization
    num_hidden_layers: Optional[int] = None

