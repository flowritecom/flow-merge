from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional

ModelId = NewType("ModelId", str)
SafetensorsIndex = Dict[str, str]

CustomId = str
Filename = str
FilePath = Path
FileToTensorIndex = Dict[str, List[str]]
HfId = str
RepoId = str
TensorKey = str
TensorIndex = Any


@dataclass(frozen=True)
class ShardFile:
    filename: str
    path: Path
    tensor_keys: Optional[List[str]] = None


ShardFiles = List[ShardFile]
