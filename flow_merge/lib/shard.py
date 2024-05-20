import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download
from safetensors import safe_open

from flow_merge.lib.constants import DeviceIdentifier
from flow_merge.lib.model_metadata import ModelMetadata

SafetensorsIndex = Dict[str, str]
RepoId = str
LocalModelId = str
LocalModelDirName = str
MaybeRepoId = str
Filename = str
HfId = str
CustomId = str
ModelFilePath = LocalModelId | RepoId


@dataclass
class ShardFile:
    filename: str
    path: str
    tensor_keys: List[str]


ShardFiles = List[ShardFile]


def get_shard_file(
    model_path: Path,
    repo_id: str,
    device: DeviceIdentifier,
    shard_filename: str,
    keys=None,
) -> ShardFile:
    output_path = os.path.join(model_path, shard_filename)
    hf_hub_download(repo_id, shard_filename, local_dir=model_path, resume_download=True)

    # we handle the case where only a single safetensors file exists, no shards
    if shard_filename == "model.safetensors":
        if os.path.exists(output_path):
            with safe_open(output_path, framework="pt", device=device) as f:
                if keys is None:
                    keys = list(f.keys())
        else:
            raise FileNotFoundError(f"File {output_path} not found.")

    return ShardFile(filename=shard_filename, path=model_path, tensor_keys=keys)


def gather_shard_files(file_to_tensor_index) -> ShardFiles:
    return [get_shard_file(filename, keys) for filename, keys in file_to_tensor_index]

    # self.md = model_source_metadata
    # self.device = device
    # self.model_path = self.md.model_absolute_path
    # self.id = self.md.id
    # self.shard_files = self.create_shard_files()


def create_shard_files(model_metadata: ModelMetadata) -> ShardFiles:
    md = model_metadata

    # first failure case
    if not (md.has_config and md.has_tokenizer_config):
        raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")
    else:
        for filename in md.file_list:
            if filename in [
                "config.json",
                "tokenizer.json",
                "tokenizer.vocab",
                "vocab.json",
                "tokenizer_config.json",
            ]:
                hf_hub_download(
                    md.repo_id, filename, local_dir=model_path, resume_download=True
                )

    # success case 1
    if md.has_safetensor_files:
        # check for config value whether to merge adapter
        if md.has_adapter:
            print("merge adapter to base model using PEFT")
            pass

        # if we have index we get the shard files
        if md.has_safetensor_index:
            return gather_shard_files()

        else:
            # if only a single safetensors file exists
            # side-effecting: downloads the single-shard, no variable assignment
            get_shard_file("model.safetensors")
            model_sf_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(model_sf_path):
                # If model.safetensors exists, assume all tensor_keys are in the same file
                with safe_open(model_sf_path, framework="pt", device=device) as f:
                    tensor_keys = list(f.keys())
                return [
                    ShardFile(
                        filename="model.safetensors",
                        path=model_path,
                        tensor_keys=tensor_keys,
                    )
                ]
