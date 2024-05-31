import json

from typing import Any, Dict
from .data_architecture.snapshot import Snapshot
from .hash import create_content_hash

class SnapshotService:
    snapshot: Snapshot = None

    def __init__(self,  env, logger):
        self.env = env
        self.logger = logger

    def create(self) -> Snapshot:
        pass

    def load(self, snapshot: Dict[str, Any]) -> None:
        pass
        # 1. run validation checks on the content hashes recursively
        # 2. load into pydantic classes - legality checks
        # 3. finally create and load into class

    def load_json(self, snapshot_json: str) -> None:
        snapshot_dict = json.loads(snapshot_json)
        return self.load(snapshot_dict)

    def to_dict(self):
        return self.snapshot.model_dump() if self._has_snapshot() else None
    
    def to_json(self):
        return self.snapshot.model_dump_json() if self._has_snapshot() else None

    def _has_snapshot(self) -> bool:
        if (self.snapshot is None):
            self.logger.warn("No existing snapshot object. Try running snapshot.create() first or load an existing snapshot using snapshot.load(snapshot_dict)")
            return False
        
        return True

    def _validate_content_hash(
            self,
            existing_hash: str,
            data_dict: Dict[str, Any]
        ) -> bool:
        content_hash = create_content_hash(data_dict)
        return content_hash == existing_hash

    def validate(self):
        """
        Integrity check for the snapshot object
        """
        pass

    def validate_key(self, key):
        pass