import json
import os
from contextlib import ExitStack
from typing import Dict, List
from pydantic import BaseModel
import huggingface_hub
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from flow_merge.lib.io import (
    has_adapter_files,
    has_config_json,
    has_pytorch_bin_files,
    has_pytorch_bin_index,
    has_safetensors_files,
    has_safetensors_index,
    has_tokenizer_config,
    has_tokenizer_file,
)
from flow_merge.lib.model import Model

    
class ShardFile(BaseModel):
    filename: str
    path: str
    tensor_keys: List[str]


SafetensorsIndex = Dict[str, str]
FileToTensorIndex = Dict[str, List[str]]
ShardFiles = List[ShardFile]
RepoId = str
LocalModelId = str
LocalModelDirName = str
MaybeRepoId = str
Filename = str

ModelFilePath = LocalModelId | RepoId


class ModelSourceDir:
    def __init__(
        self, model_id, model_abs_local_source_path, local, repo_id, file_list
    ):
        self.model_id = model_id
        self.model_abs_local_source_path = model_abs_local_source_path
        self.local = local
        self.repo_id = repo_id
        self.file_list = file_list
        self.has_config = has_config_json(self.file_list)
        self.has_vocab = has_tokenizer_file(self.file_list)
        self.has_tokenizer_config = has_tokenizer_config(self.file_list)
        self.has_safetensor_index = has_safetensors_index(self.file_list)
        self.has_pytorch_bin_index = has_pytorch_bin_index(self.file_list)
        self.has_safetensor_files = has_safetensors_files(self.file_list)
        self.has_pytorch_bin_files = has_pytorch_bin_files(self.file_list)
        self.has_adapter = has_adapter_files(self.file_list)

    # TODO: join these fns
    @staticmethod
    def get_abs_local_model_path(
        model_id, model_abs_local_source_path, merge_config
    ) -> str:
        if model_abs_local_source_path:
            return model_abs_local_source_path
        else:
            return os.path.join(merge_config.directory_settings.local_dir, model_id)

    @staticmethod
    def get_absolute_path(model_id, merge_config):
        maybe_is_path = os.path.join(
            merge_config.directory_settings.local_dir, model_id
        )
        if os.path.isdir(maybe_is_path):
            return os.path.abspath(maybe_is_path)
        else:
            return None

    @staticmethod
    def verify_repo_exists(model_id) -> str:
        try:
            repo_info = huggingface_hub.hf_api.repo_info(model_id)
            return repo_info.id
        except huggingface_hub.hf_api.RepositoryNotFoundError:
            raise RuntimeError(
                f"Model path {model_id} does not exist locally or on the Hugging Face Hub"
            )

    @staticmethod
    def get_file_list(model_id, model_abs_local_source_path, local) -> List[str]:
        if local:
            return os.listdir(model_abs_local_source_path)
        else:
            return huggingface_hub.list_repo_files(model_id, repo_type="model")

    @classmethod
    def from_model_path(
        cls, model_id: LocalModelId | MaybeRepoId, merge_config
    ) -> "ModelSourceDir":
        # we figure if model_id is a path to a local specific model directory containing the model files or a repo in HF Hub
        model_abs_local_source_path = cls.get_absolute_path(model_id, merge_config)
        local = model_abs_local_source_path is not None
        abs_local_model_path = cls.get_abs_local_model_path(
            model_id, model_abs_local_source_path, merge_config
        )
        repo_id = None if local else cls.verify_repo_exists(model_id)
        file_list = cls.get_file_list(model_id, model_abs_local_source_path, local)
        return cls(model_id, abs_local_model_path, local, repo_id, file_list)

    @staticmethod
    def get_all_warning_messages(file_list):
        all_checks = [
            (has_config_json, "Missing config.json file"),
            (has_tokenizer_config, "Missing tokenizer_config.json file"),
            (has_tokenizer_file, "Missing tokenizer vocabulary file"),
            (has_safetensors_files, "Missing .safetensors files"),
            (has_safetensors_index, "Missing model.safetensors.index.json file"),
            (has_pytorch_bin_files, "Missing pytorch_model .bin files"),
            (has_pytorch_bin_index, "Missing pytorch_model.bin.index.json file"),
            (has_adapter_files, "Missing adapter files"),
        ]
        return [warning for check, warning in all_checks if not check(file_list)]


class TensorIndex:
    def __init__(self, model_id, merge_config):
        self.device = merge_config.device
        self.md = ModelSourceDir.from_model_path(model_id, merge_config)
        self.model_path = self.md.model_abs_local_source_path
        self.repo_id = self.md.repo_id
        self.file_to_tensor_index = (
            self.create_file_to_tensor_index() if self.md.has_safetensor_index else None
        )
        self.shard_files = self.create_shard_files()

    def get_shard_file(self, filename, keys=None) -> ShardFile:
        output_path = os.path.join(self.model_path, filename)
        # TODO: check for existing shard files and their integrity and skip
        # TODO: check for in-progress downloads and resume
        if not self.md.local:
            hf_hub_download(self.repo_id, filename, local_dir=self.model_path)

        # we handle the case where only a single safetensors file exists, no shards
        if filename == "model.safetensors":
            if os.path.exists(output_path):
                with safe_open(output_path, framework="pt", device=self.device) as f:
                    if keys is None:
                        keys = list(f.keys())
            else:
                raise FileNotFoundError(f"File {output_path} not found.")

        return ShardFile(filename=filename, path=self.model_path, tensor_keys=keys)

    def gather_shard_files(self) -> ShardFiles:
        return [
            self.get_shard_file(filename, keys)
            for filename, keys in self.file_to_tensor_index
        ]

    def create_file_to_tensor_index(self) -> FileToTensorIndex:
        # fetch only if non-local
        if not self.md.local:
            file_to_tensor_index_path = hf_hub_download(
                self.repo_id, "model.safetensors.index.json", local_dir=self.model_path
            )
        else:
            file_to_tensor_index_path = os.path.join(
                self.model_path, "model.safetensors.index.json"
            )

        with open(file_to_tensor_index_path, "r") as file:
            sf_index = json.load(file)
        unique_values = {}
        for key, value in sf_index["weight_map"].items():
            unique_values.setdefault(value, []).append(key)
        return unique_values.items()

    def create_shard_files(self) -> ShardFiles:
        md = self.md

        # first failure case
        if not (md.has_config and md.has_tokenizer_config):
            raise FileNotFoundError(
                "Model is missing config.json or tokenizer_config.json"
            )
        else:
            for filename in md.file_list:
                if filename in [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer.vocab",
                    "vocab.json",
                    "tokenizer_config.json",
                    # TODO: actually check integrity and not existence
                ] and not os.path.exists(os.path.join(self.model_path, filename)):
                    hf_hub_download(md.repo_id, filename, local_dir=self.model_path)

        # success case 1
        if md.has_safetensor_files:
            # check for config value whether to merge adapter
            if md.has_adapter:
                print("merge adapter to base model using PEFT")
                pass

            # if we have index we get the shard files
            if md.has_safetensor_index:
                return self.gather_shard_files()

            else:
                # if only a single safetensors file exists
                # side-effecting: downloads the single-shard, no variable assignment
                self.get_shard_file("model.safetensors")
                model_sf_path = os.path.join(self.model_path, "model.safetensors")
                if os.path.exists(model_sf_path):
                    # If model.safetensors exists, assume all tensor_keys are in the same file
                    with safe_open(
                        model_sf_path, framework="pt", device=self.device
                    ) as f:
                        tensor_keys = list(f.keys())
                    return [
                        ShardFile(
                            filename="model.safetensors",
                            path=self.model_path,
                            tensor_keys=tensor_keys,
                        )
                    ]


class TensorLoader:
    def __init__(self, tensor_index: TensorIndex, merge_config):
        self.device = merge_config.device
        self.tensor_index = tensor_index

    def get_tensor(self, tensor_key: str) -> torch.Tensor:
        """
        Retrieves a tensor from the tensor shards based on the provided tensor key.

        Args:
            tensor_key: The key of the tensor to be retrieved. e.g. 'model.embed_tokens.weight'

        Returns:
            The tensor associated with the provided tensor key.

        Raises:
            KeyError: If the tensor key is not found in any of the tensor shards.
        """
        for shard_file in self.tensor_index.shard_files:
            if tensor_key in shard_file.tensor_keys:
                with ExitStack() as stack:
                    path_to_shard = os.path.join(shard_file.path, shard_file.filename)
                    if os.path.exists(path_to_shard):
                        file = stack.enter_context(
                            safe_open(path_to_shard, framework="pt", device=self.device)
                        )
                        return file.get_tensor(tensor_key)
                    else:
                        RuntimeError(
                            f"Path {path_to_shard} to shard file doesn't exist!"
                        )
        raise KeyError(f"Tensor key {tensor_key} not found in model {path_to_shard}")
