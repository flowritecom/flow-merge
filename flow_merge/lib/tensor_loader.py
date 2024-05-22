# # TODO: check sha, and if not match then force_download=True
# def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
#     try:
#         file_path = hf_hub_download(
#             repo_id, filename,
#             local_dir=str(local_dir),  # Convert Path to str for hf_hub_download
#             resume_download=True,
#             token=config.hf_token,
#         )
#         return Path(file_path)  # Convert returned file_path to Path

#     except FileExistsError:
#         print(f"File {local_dir / filename} already exists and is complete, skipping download.")
#         return local_dir / filename
#     except Exception as e:
#         raise RuntimeError(f"An unexpected error occurred while downloading {filename} from {repo_id}: {e}")


# def get_tensor_keys(file_path: Path, file_type: str, device: DeviceIdentifier) -> List[str]:
#     try:
#         if file_type == "safetensors":
#             with safe_open(file_path, framework="pt", device=device) as f:
#                 return list(f.keys())
#         elif file_type == "bin":
#             with open(file_path, "rb") as f:
#                 state_dict = torch.load(f, map_location=device)
#                 return list(state_dict.keys())
#         else:
#             raise ValueError(f"Unsupported file type: {file_type}")
#     except Exception as e:
#         raise RuntimeError(f"Error loading tensor keys from {file_path}: {e}")

# def create_shard_file(output_dir: Path, repo_id: str, device: DeviceIdentifier, shard_filename: str, keys: Optional[List[str]] = None) -> ShardFile:
#     output_path = output_dir / shard_filename
#     download_file(repo_id=repo_id, filename=shard_filename, local_dir=output_dir)

#     if keys is None:
#         file_type = "safetensors" if shard_filename.endswith(".safetensors") else "bin"
#         try:
#             keys = get_tensor_keys(output_path, file_type, device)
#         except RuntimeError as e:
#             keys = []  # Default to an empty list if keys cannot be retrieved

#     return ShardFile(filename=shard_filename, path=output_path, tensor_keys=keys)


# def gather_shard_files(file_to_tensor_index: Dict[str, List[str]], output_dir: Path, repo_id: str, device: DeviceIdentifier) -> List[ShardFile]:
#     try:
#         return [create_shard_file(output_dir, repo_id, device, filename, keys) for filename, keys in file_to_tensor_index.items() if is_model_file(filename)]
#     except Exception as e:
#         raise RuntimeError(f"Error gathering shard files: {e}")

# def is_model_file(filename: str) -> bool:
#     """Helper function to check if a file is a model file."""
#     return filename.endswith(('.safetensors', '.bin'))

# def download_required_files(metadata: ModelMetadata, output_dir: Path):
#     required_files = [
#         "config.json", "tokenizer.json", "tokenizer.vocab", "vocab.json", "tokenizer_config.json"
#     ]
#     for filename in required_files:
#         if filename in metadata.file_list:
#             download_file(metadata.id, filename, output_dir)

# def download_adapter_files(model_metadata: ModelMetadata, output_dir: Path):
#     adapter_files = [f for f in model_metadata.file_list if "adapter" in f]
#     for adapter_file in adapter_files:
#         download_file(model_metadata.id, adapter_file, output_dir)

# def determine_base_model_shards(model_metadata: ModelMetadata) -> List[str]:
#     if model_metadata.has_safetensor_files:
#         base_model_shards = [
#             f for f in model_metadata.file_list if f.startswith("model-") and f.endswith(".safetensors")
#         ]
#         if not base_model_shards:
#             base_model_shards = ["model.safetensors"]
#     else:
#         base_model_shards = [
#             f for f in model_metadata.file_list if f.startswith("pytorch_model-") and f.endswith(".bin")
#         ]
#         if not base_model_shards:
#             base_model_shards = ["pytorch_model.bin"]

#     return base_model_shards

# def load_and_apply_adapters(adapter_files: List[str], base_model_shards: List[str], output_dir: Path, device: DeviceIdentifier, repo_id: str) -> torch.nn.Module:
#     shard_paths = []
#     for shard_file in base_model_shards:
#         shard_path = download_file(repo_id, shard_file, output_dir)
#         shard_paths.append(shard_path)

#     peft_config = PeftConfig.from_pretrained(output_dir)
#     return PeftModel.from_pretrained(shard_paths, peft_config=peft_config)

# def save_model_shards(base_model: torch.nn.Module, output_dir: Path) -> List[ShardFile]:
#     shard_files = []
#     with TensorWriter(output_dir) as writer:
#         for name, param in base_model.named_parameters():
#             writer.save_tensor(ModelWeight(name=name), param)
#             shard_files.append(ShardFile(filename=name, path=str(output_dir)))
#         writer.finish()
#     return shard_files

# def merge_and_save_model(model_metadata: ModelMetadata, output_dir: Path, device: DeviceIdentifier) -> List[ShardFile]:
#     download_adapter_files(model_metadata, output_dir)
#     base_model_shards = determine_base_model_shards(model_metadata)
#     adapter_files = [f for f in model_metadata.file_list if "adapter" in f]

#     base_model = load_and_apply_adapters(
#         adapter_files=adapter_files,
#         base_model_shards=base_model_shards,
#         output_dir=output_dir,
#         device=device,
#         repo_id=model_metadata.id
#     )
#     return save_model_shards(base_model, output_dir)

# def create_shard_files(
#     model_metadata: ModelMetadata,
#     device: DeviceIdentifier,
#     directory_settings: DirectorySettings = DirectorySettings(),
# ) -> List[ShardFile]:
#     output_dir = directory_settings.output_dir / model_metadata.id

#     if not (model_metadata.has_config and model_metadata.has_tokenizer_config):
#         raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")

#     download_required_files(model_metadata, output_dir)

#     if model_metadata.has_adapter:
#         return merge_and_save_model(model_metadata, output_dir, device)

#     file_index = TensorIndexService.create_file_to_tensor_index(
#         model_metadata.hf_exists, model_metadata.id, output_dir
#     )
#     if file_index:
#         return gather_shard_files(file_index, output_dir, model_metadata.id, device)
#     else:
#         print("Index files not found, using single shard file fallback.")
#         single_file = "model.safetensors" if model_metadata.has_safetensor_files else "pytorch_model.bin"
#         shard_file = create_shard_file(output_dir, model_metadata.id, device, single_file)
#         return [shard_file]


# class TensorLoader(BaseModel, arbitrary_types_allowed=True):
#     model: Any
#     device: torch.device

#     class Config:
#         frozen = True

#     def get_tensor(self, tensor_key: TensorKey) -> torch.Tensor:
#         """
#         Retrieves a tensor from the tensor shards based on the provided tensor key.

#         Args:
#             tensor_key: The key of the tensor to be retrieved. e.g. 'model.embed_tokens.weight'

#         Returns:
#             The tensor associated with the provided tensor key.

#         Raises:
#             KeyError: If the tensor key is not found in any of the tensor shards.
#         """
#         for shard_file in self.model.shards:
#             if tensor_key in shard_file.tensor_keys:
#                 return self._load_tensor_from_shard(shard_file, tensor_key)
#         raise KeyError(f"Tensor key {tensor_key} not found in model {self.model.path}")


#     def _load_tensor_from_shard(
#         self, shard_file: ShardFile, tensor_key: TensorKey
#     ) -> torch.Tensor:
#         path_to_shard = shard_file.path
#         if not path_to_shard.exists():
#             raise RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")

#         if path_to_shard.suffix == ".safetensors":
#             return self._load_safetensor(path_to_shard, tensor_key)
#         elif path_to_shard.suffix == ".bin":
#             return self._load_bin_tensor(path_to_shard, tensor_key)
#         else:
#             raise ValueError(f"Unsupported file type: {path_to_shard.suffix}")


#     def _load_safetensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
#         with safe_open(path, framework="pt", device=self.device) as file:
#             return file.get_tensor(tensor_key)

#     def _load_bin_tensor(self, path: FilePath, tensor_key: TensorKey) -> torch.Tensor:
#         with path.open("rb") as f:
#             state_dict = torch.load(f, map_location=self.device)
#             if tensor_key in state_dict:
#                 return state_dict[tensor_key]
#             raise KeyError(f"Tensor key {tensor_key} not found in file {path.name}")


import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field
import torch
from safetensors import safe_open

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.constants import DeviceIdentifier
from flow_merge.lib.model_metadata import ModelMetadata
from flow_merge.lib.config import config
from flow_merge.lib.merge_settings import DirectorySettings
from peft import PeftModel, PeftConfig

import json
from pathlib import Path
from typing import Dict, List, NewType, Optional

from flow_merge.lib.tensor_writer import TensorWriter

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from flow_merge.lib.config import config
import torch
from safetensors import safe_open
from pathlib import Path
from typing import List
from flow_merge.lib.constants import DeviceIdentifier
from pathlib import Path
from typing import List, Optional, Dict
import torch
from peft import PeftModel, PeftConfig
from flow_merge.lib.model_metadata import ModelMetadata
from flow_merge.lib.tensor_writer import TensorWriter
from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.constants import DeviceIdentifier
from flow_merge.lib.config import config
from typing import Any
import torch

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


class FileRepository:
    """Immutable repository for handling file operations."""

    @staticmethod
    def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
        try:
            file_path = hf_hub_download(
                repo_id,
                filename,
                local_dir=str(local_dir),  # Convert Path to str for hf_hub_download
                resume_download=True,
                token=config.hf_token,
            )
            return Path(file_path)  # Convert returned file_path to Path

        except FileExistsError:
            print(
                f"File {local_dir / filename} already exists and is complete, skipping download."
            )
            return local_dir / filename
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while downloading {filename} from {repo_id}: {e}"
            )

    @staticmethod
    def load_index(file_path: Path) -> dict:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error loading index from {file_path}: {e}")

    @staticmethod
    def download_required_files(metadata: "ModelMetadata"):
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer.vocab",
            "vocab.json",
            "tokenizer_config.json",
        ]
        for filename in required_files:
            if filename in metadata.file_list:
                FileRepository.download_file(
                    metadata.id, filename, metadata.directory_settings.local_dir
                )

    @staticmethod
    def download_adapter_files(model_metadata: "ModelMetadata"):
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]
        for adapter_file in adapter_files:
            FileRepository.download_file(
                model_metadata.id,
                adapter_file,
                model_metadata.directory_settings.local_dir,
            )


class TensorOperations:
    """Handles tensor-related operations."""

    @staticmethod
    def load_safetensor(
        path: Path, tensor_key: str, device: torch.device
    ) -> torch.Tensor:
        with safe_open(path, framework="pt", device=device) as file:
            return file.get_tensor(tensor_key)

    @staticmethod
    def load_bin_tensor(
        path: Path, tensor_key: str, device: torch.device
    ) -> torch.Tensor:
        with path.open("rb") as f:
            state_dict = torch.load(f, map_location=device)
            if tensor_key in state_dict:
                return state_dict[tensor_key]
            raise KeyError(f"Tensor key {tensor_key} not found in file {path.name}")

    @staticmethod
    def get_tensor_keys(
        file_path: Path, file_type: str, device: DeviceIdentifier
    ) -> List[str]:
        try:
            if file_type == "safetensors":
                with safe_open(file_path, framework="pt", device=device) as f:
                    return list(f.keys())
            elif file_type == "bin":
                with open(file_path, "rb") as f:
                    state_dict = torch.load(f, map_location=device)
                    return list(state_dict.keys())
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise RuntimeError(f"Error loading tensor keys from {file_path}: {e}")


class TensorIndexService:
    """Service for handling tensor index operations."""

    @staticmethod
    def flip_keys(shardfile_index: Dict[str, str]) -> Dict[str, list]:
        unique_values = {}
        for key, value in shardfile_index["weight_map"].items():
            unique_values.setdefault(value, []).append(key)
        return unique_values

    @staticmethod
    def create_file_to_tensor_index(
        metadata: ModelMetadata,
    ) -> Optional[Dict[str, list]]:
        index_path = None
        model_path = metadata.absolute_path

        if metadata.hf_exists and metadata.has_safetensors_index:
            try:
                index_path = FileRepository.download_file(
                    repo_id=str(metadata.id),
                    filename="model.safetensors.index.json",
                    local_dir=metadata.directory_settings.local_dir,
                )
            except Exception as e:
                print(f"Safetensors index not found: {e}")

        elif metadata.hf_exists and metadata.has_pytorch_bin_index:
            try:
                index_path = FileRepository.download_file(
                    repo_id=str(metadata.id),
                    filename="pytorch_model.bin.index.json",
                    local_dir=metadata.directory_settings.local_dir,
                )
            except Exception as e:
                print(f"Pytorch bin index not found: {e}")
                return None
        else:
            safetensors_index_path = Path(model_path) / "model.safetensors.index.json"
            pytorch_bin_index_path = Path(model_path) / "pytorch_model.bin.index.json"

            if safetensors_index_path.exists():
                index_path = safetensors_index_path
            elif pytorch_bin_index_path.exists():
                index_path = pytorch_bin_index_path
            else:
                return None

        shardfile_index = FileRepository.load_index(index_path)
        return TensorIndexService.flip_keys(shardfile_index)


class ModelService:
    """Manages the overall process of handling models."""

    @staticmethod
    def create_shard_file(
        output_dir: Path,
        repo_id: str,
        device: DeviceIdentifier,
        shard_filename: str,
        keys: Optional[List[str]] = None,
    ) -> ShardFile:
        output_path = output_dir / shard_filename
        FileRepository.download_file(
            repo_id=repo_id, filename=shard_filename, local_dir=output_dir
        )

        if keys is None:
            file_type = (
                "safetensors" if shard_filename.endswith(".safetensors") else "bin"
            )
            try:
                keys = TensorOperations.get_tensor_keys(output_path, file_type, device)
            except RuntimeError as e:
                keys = []  # Default to an empty list if keys cannot be retrieved

        return ShardFile(filename=shard_filename, path=output_path, tensor_keys=keys)

    @staticmethod
    def gather_shard_files(
        file_to_tensor_index: Dict[str, List[str]],
        output_dir: Path,
        repo_id: str,
        device: DeviceIdentifier,
    ) -> List[ShardFile]:
        try:
            return [
                ModelService.create_shard_file(
                    output_dir, repo_id, device, filename, keys
                )
                for filename, keys in file_to_tensor_index.items()
                if filename.endswith((".safetensors", ".bin"))
            ]
        except Exception as e:
            raise RuntimeError(f"Error gathering shard files: {e}")

    @staticmethod
    def determine_base_model_shards(model_metadata: ModelMetadata) -> List[str]:
        if model_metadata.has_safetensor_files:
            base_model_shards = [
                f
                for f in model_metadata.file_list
                if f.startswith("model-") and f.endswith(".safetensors")
            ]
            if not base_model_shards:
                base_model_shards = ["model.safetensors"]
        else:
            base_model_shards = [
                f
                for f in model_metadata.file_list
                if f.startswith("pytorch_model-") and f.endswith(".bin")
            ]
            if not base_model_shards:
                base_model_shards = ["pytorch_model.bin"]

        return base_model_shards

    @staticmethod
    def load_and_apply_adapters(
        adapter_files: List[str],
        base_model_shards: List[str],
        device: DeviceIdentifier,
        repo_id: str,
        local_dir: Path,
    ) -> torch.nn.Module:
        shard_paths = []
        for shard_file in base_model_shards:
            shard_path = FileRepository.download_file(repo_id, shard_file, local_dir)
            shard_paths.append(shard_path)

        peft_config = PeftConfig.from_pretrained(local_dir)
        return PeftModel.from_pretrained(shard_paths, peft_config=peft_config)

    @staticmethod
    def save_model_shards(
        base_model: torch.nn.Module, output_dir: Path
    ) -> List[ShardFile]:
        shard_files = []
        with TensorWriter(output_dir) as writer:
            for name, param in base_model.named_parameters():
                writer.save_tensor(ModelWeight(name=name), param)
                shard_files.append(ShardFile(filename=name, path=str(output_dir)))
            writer.finish()
        return shard_files

    @staticmethod
    def merge_and_save_model(
        model_metadata: ModelMetadata, device: DeviceIdentifier
    ) -> List[ShardFile]:
        FileRepository.download_adapter_files(model_metadata)
        base_model_shards = ModelService.determine_base_model_shards(model_metadata)
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]

        base_model = ModelService.load_and_apply_adapters(
            adapter_files=adapter_files,
            base_model_shards=base_model_shards,
            device=device,
            repo_id=model_metadata.id,
            local_dir=model_metadata.directory_settings.local_dir,
        )
        return ModelService.save_model_shards(
            base_model, model_metadata.directory_settings.output_dir
        )

    @staticmethod
    def create_shard_files(
        model_metadata: ModelMetadata, device: DeviceIdentifier
    ) -> List[ShardFile]:
        output_dir = model_metadata.directory_settings.output_dir / model_metadata.id

        if not (model_metadata.has_config and model_metadata.has_tokenizer_config):
            raise FileNotFoundError(
                "Model is missing config.json or tokenizer_config.json"
            )

        FileRepository.download_required_files(model_metadata)

        if model_metadata.has_adapter:
            return ModelService.merge_and_save_model(model_metadata, device)

        file_index = TensorIndexService.create_file_to_tensor_index(model_metadata)
        if file_index:
            return ModelService.gather_shard_files(
                file_index, output_dir, model_metadata.id, device
            )
        else:
            print("Index files not found, using single shard file fallback.")
            single_file = (
                "model.safetensors"
                if model_metadata.has_safetensor_files
                else "pytorch_model.bin"
            )
            shard_file = ModelService.create_shard_file(
                output_dir, model_metadata.id, device, single_file
            )
            return [shard_file]


class TensorLoader(BaseModel, arbitrary_types_allowed=True):
    model: Any
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
        path_to_shard = shard_file.path
        if not path_to_shard.exists():
            raise RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")

        if path_to_shard.suffix == ".safetensors":
            return TensorOperations.load_safetensor(
                path_to_shard, tensor_key, self.device
            )
        elif path_to_shard.suffix == ".bin":
            return TensorOperations.load_bin_tensor(
                path_to_shard, tensor_key, self.device
            )
        else:
            raise ValueError(f"Unsupported file type: {path_to_shard.suffix}")
