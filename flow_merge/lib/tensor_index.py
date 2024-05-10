import json
import os
from typing import Dict, List
from huggingface_hub import hf_hub_download
from flow_merge.lib.shard import ShardFile

SafetensorsIndex = Dict[str, str]
FileToTensorIndex = Dict[str, List[str]]
ShardFiles = List[ShardFile]

def flip_keys(shardfile_index):
    unique_values = {}
    for key, value in shardfile_index["weight_map"].items():
        unique_values.setdefault(value, []).append(key)
    return unique_values.items()


# have this check located outside of the function
#if self.md.has_safetensor_index else None

def create_file_to_tensor_index(hf_exists, repo_id, model_path) -> FileToTensorIndex:

    # locate the file (download if necessary)
    if hf_exists:
        file_to_tensor_index_path = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            local_dir=model_path,
            resume_download=True,
        )
    else:
        file_to_tensor_index_path = model_path / "model.safetensors.index.json"

    # load the file
    with open(file_to_tensor_index_path, "r") as file:
        shardfile_index = json.load(file)
    
    # flip the shard file values to be keys and the tensors to be values
    result = flip_keys(shardfile_index)

    return result