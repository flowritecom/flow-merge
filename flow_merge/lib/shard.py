import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModel
from peft import PeftModel, PeftConfig

from flow_merge.lib.constants import DeviceIdentifier
from flow_merge.lib.model_metadata import ModelMetadata
from flow_merge.lib.config import config
from flow_merge.lib.merge_settings import DirectorySettings

RepoId = str
Filename = str
TensorKeys = List[str]


@dataclass(frozen=True)
class ShardFile:
    filename: Filename
    path: str
    tensor_keys: TensorKeys


ShardFiles = List[ShardFile]


def download_file(repo_id: RepoId, filename: Filename, local_dir: Path) -> str:
    return hf_hub_download(repo_id, filename, local_dir=local_dir, resume_download=True, token=config.hf_token)


def get_shard_file(
    model_path: Path,
    repo_id: RepoId,
    device: DeviceIdentifier,
    shard_filename: Filename,
    keys: Optional[TensorKeys] = None,
) -> ShardFile:
    output_path = model_path / shard_filename
    download_file(repo_id, shard_filename, model_path)

    if shard_filename == "model.safetensors":
        if output_path.exists():
            with safe_open(output_path, framework="pt", device=device) as f:
                if keys is None:
                    keys = list(f.keys())
        else:
            raise FileNotFoundError(f"File {output_path} not found.")

    return ShardFile(filename=shard_filename, path=str(model_path), tensor_keys=keys)


def gather_shard_files(
    file_to_tensor_index: Dict[Filename, TensorKeys],
    model_path: Path,
    repo_id: RepoId,
    device: DeviceIdentifier
) -> ShardFiles:
    return [get_shard_file(model_path, repo_id, device, filename, keys) for filename, keys in file_to_tensor_index.items()]


def download_required_files(metadata: ModelMetadata, model_path: Path):
    required_files: List[Filename] = [
        "config.json",
        "tokenizer.json",
        "tokenizer.vocab",
        "vocab.json",
        "tokenizer_config.json",
    ]
    for filename in required_files:
        if filename in metadata.file_list:
            download_file(metadata.id, filename, model_path)


def handle_safetensor_files(
    model_metadata: ModelMetadata, model_path: Path, device: DeviceIdentifier
) -> ShardFiles:
    if model_metadata.has_safetensor_index:
        return gather_shard_files(model_metadata.file_metadata_list, model_path, model_metadata.id, device)
    else:
        shard_file = get_shard_file(model_path, model_metadata.id, device, "model.safetensors")
        model_sf_path = model_path / "model.safetensors"
        if model_sf_path.exists():
            with safe_open(model_sf_path, framework="pt", device=device) as f:
                tensor_keys = list(f.keys())
            return [
                ShardFile(
                    filename="model.safetensors",
                    path=str(model_path),
                    tensor_keys=tensor_keys,
                )
            ]
    raise FileNotFoundError("Model does not contain safetensor files.")


def handle_pytorch_bin_files(
    model_metadata: ModelMetadata, model_path: Path, device: DeviceIdentifier
) -> ShardFiles:
    shard_files: ShardFiles = []
    for filename in model_metadata.file_list:
        if filename.endswith(".bin"):
            download_file(model_metadata.id, filename, model_path)
            shard_files.append(ShardFile(filename=filename, path=str(model_path), tensor_keys=[]))
    return shard_files


def merge_adapter_with_base_model(
    model_metadata: ModelMetadata, model_path: Path, device: DeviceIdentifier
) -> ShardFile:
    adapter_files: List[Filename] = [f for f in model_metadata.file_list if "adapter" in f]
    for adapter_file in adapter_files:
        download_file(model_metadata.id, adapter_file, model_path)

    base_model_path = model_path / "model.safetensors"
    if not base_model_path.exists():
        base_model_path = model_path / "pytorch_model.bin"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model file {base_model_path} not found.")

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = PeftModel.from_pretrained(base_model_path, peft_config=peft_config)
    merged_model_path = model_path / "merged_model.safetensors"
    base_model.save_pretrained(merged_model_path)

    return ShardFile(
        filename="merged_model.safetensors",
        path=str(model_path),
        tensor_keys=[]
    )


def create_shard_files(
    model_metadata: ModelMetadata, device: DeviceIdentifier, directory_settings: DirectorySettings = DirectorySettings()
) -> ShardFiles:
    md = model_metadata
    model_path = directory_settings.local_dir / md.id

    if not (md.has_config and md.has_tokenizer_config):
        raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")

    download_required_files(md, model_path)

    if md.has_safetensor_files:
        if md.has_adapter:
            return [merge_adapter_with_base_model(md, model_path, device)]
        return handle_safetensor_files(md, model_path, device)

    if md.has_pytorch_bin_files:
        if md.has_adapter:
            return [merge_adapter_with_base_model(md, model_path, device)]
        return handle_pytorch_bin_files(md, model_path, device)

    raise FileNotFoundError("Model does not contain safetensor or pytorch bin files.")
