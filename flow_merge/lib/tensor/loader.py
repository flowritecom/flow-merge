from pathlib import Path
from typing import List, Optional
import torch
from pydantic import BaseModel
from safetensors import safe_open
from flow_merge.lib.config import DeviceIdentifier

TensorKey = str


class ShardFile(BaseModel):
    filename: str
    path: Path
    tensor_keys: Optional[List[str]] = None

    class Config:
        frozen = True


class TensorRepository:
    """Handles tensor-related operations and retrieval."""

    @staticmethod
    def get_tensor(
            shards: List[ShardFile], tensor_key: TensorKey, device: DeviceIdentifier
    ) -> torch.Tensor:
        """
        Retrieves a tensor from the tensor shards based on the provided tensor key.
        Args:
            shards: List of shard files to search for the tensor.
            tensor_key: The key of the tensor to be retrieved. e.g. 'model.embed_tokens.weight'
            device: The device to load the tensor onto.
        Returns:
            The tensor associated with the provided tensor key.
        Raises:
            KeyError: If the tensor key is not found in any of the tensor shards.
        """
        for shard_file in shards:
            if shard_file.tensor_keys and tensor_key in shard_file.tensor_keys:
                return TensorRepository.load_tensor(shard_file, tensor_key, device)
        raise KeyError(f"Tensor key {tensor_key} not found in provided shards.")

    @staticmethod
    def load_tensor(
            shard_file: ShardFile, tensor_key: TensorKey, device: DeviceIdentifier
    ) -> torch.Tensor:
        """
        Load a tensor from a specific shard file (either .safetensors or .bin).
        Args:
            shard_file: The shard file containing the tensor.
            tensor_key: The key of the tensor to be loaded.
            device: The device to load the tensor onto.
        Returns:
            The loaded tensor.
        Raises:
            RuntimeError: If the shard file path does not exist.
            ValueError: If the file type is unsupported.
        """
        path_to_shard = shard_file.path / shard_file.filename
        if not path_to_shard.exists():
            raise RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")

        if path_to_shard.suffix == ".safetensors":
            return TensorRepository._load_safetensor(path_to_shard, tensor_key, device)
        elif path_to_shard.suffix == ".bin":
            return TensorRepository._load_bin_tensor(path_to_shard, tensor_key, device)
        else:
            raise ValueError(f"Unsupported file type: {path_to_shard.suffix}")

    @staticmethod
    def _load_safetensor(
            path: Path, tensor_key: str, device: DeviceIdentifier
    ) -> torch.Tensor:
        """
        Load a tensor from a safetensor file.
        Args:
            path: The path to the safetensor file.
            tensor_key: The key of the tensor to be loaded.
            device: The device to load the tensor onto.
        Returns:
            The loaded tensor.
        """
        with safe_open(path, framework="pt", device=device.value) as file:
            return file.get_tensor(tensor_key)

    @staticmethod
    def _load_bin_tensor(
            path: Path, tensor_key: str, device: DeviceIdentifier
    ) -> torch.Tensor:
        """
        Load a tensor from a binary file.
        Args:
            path: The path to the binary file.
            tensor_key: The key of the tensor to be loaded.
            device: The device to load the tensor onto.
        Returns:
            The loaded tensor.
        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        with path.open("rb") as f:
            state_dict = torch.load(f, map_location=device.value)
            if tensor_key in state_dict:
                return state_dict[tensor_key]
            raise KeyError(f"Tensor key {tensor_key} not found in file {path.name}")

    @staticmethod
    def get_tensor_keys_from_file(
            file_path: Path, device: DeviceIdentifier
    ) -> List[str]:
        """
        Get tensor keys from a file.
        Args:
            file_path: The path to the file.
            file_type: The type of the file (safetensors or bin).
            device: The device identifier.
        Returns:
            A list of tensor keys.
        Raises:
            ValueError: If the file type is unsupported.
            RuntimeError: If there is an error loading tensor keys from the file.
        """
        try:
            if file_path.suffix.endswith("safetensors"):
                with safe_open(file_path, framework="pt", device=device.value) as f:
                    return list(f.keys())
            elif file_path.suffix.endswith("bin"):
                with open(file_path, "rb") as f:
                    state_dict = torch.load(f, map_location=device.value)
                    return list(state_dict.keys())
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            raise RuntimeError(f"Error loading tensor keys from {file_path}: {e}")
