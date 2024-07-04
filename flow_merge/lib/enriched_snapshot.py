from typing import List, Any

from flow_merge.lib.snapshot.data_architecture.snapshot import Snapshot
from flow_merge.lib.model.model import Model

class EnrichedSnapshot(Snapshot):
    models: List[Model]
    base_model: Model
    tokenizer: Any = None
