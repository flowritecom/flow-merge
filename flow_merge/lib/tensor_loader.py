import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field
import torch
from safetensors import safe_open

from flow_merge.lib.model import Model
from flow_merge.lib.shard import ShardFile

CustomId = str
Filename = str
FilePath = Path
FileToTensorIndex = Dict[str, List[str]]
HfId = str
RepoId = str
ShardFiles = List[ShardFile]
TensorKey = str
TensorIndex = Any


class TensorLoader(BaseModel):
    model: Model
    device: torch.device

    class Config:
        frozen = True

    def get_tensor(self, tensor_key: TensorKey) -> torch.Tensor:
        """
        Retrieves a tensor from the tensor shards based on the provided tensor key.

        Args:
            tensor_key: The key of the tensor to be retrieved. e.g. 'model.embed_tokens.weight'

        Returns:
            The tensor associated with the provided tensor key.

        Raises:
            KeyError: If the tensor key is not found in any of the tensor shards.
        """
        for shard_file in self.model.shards:
            if tensor_key in shard_file.tensor_keys:
                return self._load_tensor_from_shard(shard_file, tensor_key)
        raise KeyError(f"Tensor key {tensor_key} not found in model {self.model.path}")

    def _load_tensor_from_shard(
        self, shard_file: ShardFile, tensor_key: TensorKey
    ) -> torch.Tensor:
        path_to_shard = shard_file.path / shard_file.filename
        if not path_to_shard.exists():
            raise RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")

        if path_to_shard.suffix == ".safetensors":
            return self._load_safetensor(path_to_shard, tensor_key)
        elif path_to_shard.suffix == ".bin":
            return self._load_bin_tensor(path_to_shard, tensor_key)
        else:
            raise ValueError(f"Unsupported file type: {path_to_shard.suffix}")

    def _load_safetensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
        with safe_open(path, framework="pt", device=self.device) as file:
            return file.get_tensor(tensor_key)

    def _load_bin_tensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
        with path.open("rb") as f:
            state_dict = torch.load(f, map_location=self.device)
            if tensor_key in state_dict:
                return state_dict[tensor_key]
            raise KeyError(f"Tensor key {tensor_key} not found in file {path.name}")
