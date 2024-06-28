from typing import List

from flow_merge.lib.snapshot import Snapshot
from flow_merge.lib.model.model import Model
from flow_merge.lib.tokenizer import Tokenizer

class EnrichedSnapshot(Snapshot):
    models: List[Model]
    base_model: Model
    tokenizer: Tokenizer = None
