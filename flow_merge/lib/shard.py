from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from peft import PeftModel, PeftConfig
import torch

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
    try:
        return hf_hub_download(repo_id, filename, local_dir=local_dir, resume_download=True, token=config.hf_token)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {e}")


def get_shard_file(
    output_dir: Path,
    repo_id: RepoId,
    device: DeviceIdentifier,
    shard_filename: Filename,
    keys: Optional[TensorKeys] = None,
) -> ShardFile:
    output_path = output_dir / shard_filename
    download_file(repo_id, shard_filename, output_dir)

    if shard_filename.endswith(".safetensors"):
        if output_path.exists():
            with safe_open(output_path, framework="pt", device=device) as f:
                if keys is None:
                    keys = list(f.keys())
        else:
            raise FileNotFoundError(f"File {output_path} not found.")

    return ShardFile(filename=shard_filename, path=str(output_dir), tensor_keys=keys)


def gather_shard_files(
    file_to_tensor_index: Dict[Filename, TensorKeys],
    output_dir: Path,
    repo_id: RepoId,
    device: DeviceIdentifier
) -> ShardFiles:
    return [get_shard_file(output_dir, repo_id, device, filename, keys) for filename, keys in file_to_tensor_index.items()]


def download_required_files(metadata: ModelMetadata, output_dir: Path):
    required_files: List[Filename] = [
        "config.json",
        "tokenizer.json",
        "tokenizer.vocab",
        "vocab.json",
        "tokenizer_config.json",
    ]
    for filename in required_files:
        if filename in metadata.file_list:
            download_file(metadata.id, filename, output_dir)


def handle_safetensor_files(
    model_metadata: ModelMetadata, output_dir: Path, device: DeviceIdentifier
) -> ShardFiles:
    if model_metadata.has_safetensor_index:
        return gather_shard_files(model_metadata.file_metadata_list, output_dir, model_metadata.id, device)
    else:
        get_shard_file(output_dir, model_metadata.id, device, "model.safetensors")
        model_sf_path = output_dir / "model.safetensors"
        if model_sf_path.exists():
            with safe_open(model_sf_path, framework="pt", device=device) as f:
                tensor_keys = list(f.keys())
            return [
                ShardFile(
                    filename="model.safetensors",
                    path=str(output_dir),
                    tensor_keys=tensor_keys,
                )
            ]
    raise FileNotFoundError("Model does not contain safetensor files.")


def handle_pytorch_bin_files(
    model_metadata: ModelMetadata, output_dir: Path, device: DeviceIdentifier
) -> ShardFiles:
    shard_files: ShardFiles = []
    for filename in model_metadata.file_list:
        if filename.endswith(".bin"):
            download_file(model_metadata.id, filename, output_dir)
            bin_file_path = output_dir / filename
            if not bin_file_path.exists():
                raise FileNotFoundError(f"File {bin_file_path} not found.")
            
            state_dict = torch.load(bin_file_path, map_location=device)
            tensor_keys = list(state_dict.keys())
            
            shard_files.append(ShardFile(filename=filename, path=str(output_dir), tensor_keys=tensor_keys))
    return shard_files


def merge_adapter_with_base_model(
    model_metadata: ModelMetadata, output_dir: Path, device: DeviceIdentifier
) -> ShardFile:
    adapter_files: List[Filename] = [f for f in model_metadata.file_list if "adapter" in f]
    for adapter_file in adapter_files:
        download_file(model_metadata.id, adapter_file, output_dir)

    if model_metadata.has_safetensor_files:
        base_model_shards = [f for f in model_metadata.file_list if f.startswith("model-") and f.endswith(".safetensors")]
        if not base_model_shards:
            base_model_shards = ["model.safetensors"]
    else:
        base_model_shards = [f for f in model_metadata.file_list if f.startswith("pytorch_model-") and f.endswith(".bin")]
        if not base_model_shards:
            base_model_shards = ["pytorch_model.bin"]

    shard_paths = []
    for shard_file in base_model_shards:
        download_file(model_metadata.id, shard_file, output_dir)
        shard_path = output_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Base model shard file {shard_path} not found.")
        shard_paths.append(str(shard_path))
    
    peft_config = PeftConfig.from_pretrained(output_dir)
    base_model = PeftModel.from_pretrained(shard_paths, peft_config=peft_config)
    merged_model_path = output_dir / "merged_model.safetensors"
    base_model.save_pretrained(merged_model_path)

    return ShardFile(
        filename="merged_model.safetensors",
        path=str(output_dir),
        tensor_keys=[]
    )


def create_shard_files(
    model_metadata: ModelMetadata, device: DeviceIdentifier, directory_settings: DirectorySettings = DirectorySettings()
) -> ShardFiles:
    md = model_metadata
    output_dir = directory_settings.output_dir / md.id

    if not (md.has_config and md.has_tokenizer_config):
        raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")

    download_required_files(md, output_dir)

    if md.has_safetensor_files:
        if md.has_adapter:
            return [merge_adapter_with_base_model(md, output_dir, device)]
        return handle_safetensor_files(md, output_dir, device)

    if md.has_pytorch_bin_files:
        if md.has_adapter:
            return [merge_adapter_with_base_model(md, output_dir, device)]
        return handle_pytorch_bin_files(md, output_dir, device)

    raise FileNotFoundError("Model does not contain safetensor or pytorch bin files.")
