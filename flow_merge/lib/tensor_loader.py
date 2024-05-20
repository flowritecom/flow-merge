import json
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import huggingface_hub
from pydantic import BaseModel, field_validator
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from huggingface_hub.hf_api import (
    ModelInfo,
    RepoSibling,
    SafeTensorsInfo,
    TransformersInfo,
)

from flow_merge.lib.constants import DeviceIdentifier
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
from flow_merge.lib.merge_settings import DirectorySettings
from flow_merge.lib.model import Model
from flow_merge.lib.shard import ShardFile


SafetensorsIndex = Dict[str, str]
FileToTensorIndex = Dict[str, List[str]]
ShardFiles = List[ShardFile]
RepoId = str
LocalModelId = str
LocalModelDirName = str
MaybeRepoId = str
Filename = str

HfId = str
CustomId = str

ModelFilePath = LocalModelId | RepoId


def get_tensor(
    self, tensor_index: TensorIndex, tensor_key: str, device=device
) -> torch.Tensor:
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
                    RuntimeError(f"Path {path_to_shard} to shard file doesn't exist!")
    raise KeyError(f"Tensor key {tensor_key} not found in model {path_to_shard}")
