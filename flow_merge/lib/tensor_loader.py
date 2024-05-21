import os
from typing import Dict, List

from pydantic import BaseModel, Field
import torch
from safetensors import safe_open

from flow_merge.lib.model import Model
from flow_merge.lib.shard import ShardFile

CustomId = str
Filename = str
FilePath = str
FileToTensorIndex = Dict[str, List[str]]
HfId = str
RepoId = str
SafetensorsIndex = Dict[str, str]
ShardFiles = List[ShardFile]
TensorKey = str


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

    def _load_tensor_from_shard(self, shard_file: ShardFile, tensor_key: TensorKey) -> torch.Tensor:
        path_to_shard = os.path.join(shard_file.path, shard_file.filename)
        if not os.path.exists(path_to_shard):
            raise RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")

        if shard_file.filename.endswith(".safetensors"):
            return self._load_safetensor(path_to_shard, tensor_key)
        elif shard_file.filename.endswith(".bin"):
            return self._load_bin_tensor(path_to_shard, tensor_key)
        else:
            raise ValueError(f"Unsupported file type: {shard_file.filename}")

    def _load_safetensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
        with safe_open(path, framework="pt", device=self.device) as file:
            return file.get_tensor(tensor_key)

    def _load_bin_tensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location=self.device)
            if tensor_key in state_dict:
                return state_dict[tensor_key]
            raise KeyError(f"Tensor key {tensor_key} not found in file {os.path.basename(path)}")
