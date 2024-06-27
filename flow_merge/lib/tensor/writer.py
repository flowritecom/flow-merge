import json
import os
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from flow_merge.lib.logger import get_logger

logger = get_logger(__name__)


# TODO - Test this class with .bin files
class TensorWriter:
    def __init__(
        self,
        output_dir: Path,
        max_shard_size: int = 1000 * 1000 * 1000 * 2,
        safe_serialization: bool = True,
    ) -> None:
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.output_dir = output_dir
        self.shard_count = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._cleanup_shards()
        return False  # Propagate the exception

    def _cleanup_shards(self):
        for shard_name in set(self.weight_map.values()):
            shard_path = os.path.join(self.output_dir, shard_name)
            if os.path.exists(shard_path):
                logger.info(f"Removing shard {shard_name}")
                os.remove(shard_path)

    def save_tensor(self, weight: Any, tensor: torch.Tensor, clone: bool = False):
        if clone:
            tensor = tensor.clone()

        self.current_shard[weight.name] = tensor
        self.current_shard_size += tensor.numel() * tensor.element_size()

        if self.current_shard_size > self.max_shard_size:
            self.current_shard_to_disk()

    def current_shard_to_disk(self):
        if not self.current_shard:
            return

        self.shard_count += 1
        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{self.shard_count:05d}{extension}"
        shard_path = os.path.join(self.output_dir, shard_name)

        for weight_name in self.current_shard:
            self.weight_map[weight_name] = shard_name

        logger.info(f"Writing shard {shard_name} to disk")

        if self.safe_serialization:
            self._save_st(shard_path)
        else:
            torch.save(self.current_shard, shard_path)

        self.current_shard = {}
        self.current_shard_size = 0

    def finish(self):
        self.current_shard_to_disk()

        prefix, extension = self._get_name_components()
        shard_names = set(self.weight_map.values())
        total_shards = len(shard_names)

        for shard_name in shard_names:
            old_path = os.path.join(self.output_dir, shard_name)
            shard_num = str(shard_name.split("-")[1].split(".")[0])
            new_shard_name = f"{prefix}-{shard_num}-of-{total_shards:05d}{extension}"
            new_path = os.path.join(self.output_dir, new_shard_name)
            os.rename(old_path, new_path)

            for weight_name, mapped_shard_name in self.weight_map.items():
                if mapped_shard_name == shard_name:
                    self.weight_map[weight_name] = new_shard_name

        index_file = f"{prefix}.safetensors.index.json"
        index_path = os.path.join(self.output_dir, index_file)
        logger.info(f"Writing index file: {index_file} to disk")

        with open(index_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "total_shards": total_shards,
                        "flow-ai-merger_version": "0.1.0",
                    },
                    "weight_map": self.weight_map,
                },
                f,
            )

    def _get_name_components(self):
        if self.safe_serialization:
            return "model", ".safetensors"
        else:
            return "pytorch_model", ".bin"

    def _save_st(self, shard_path: str):
        try:
            safetensors.torch.save_file(
                self.current_shard, shard_path, metadata={"format": "pt"}
            )
        except RuntimeError as e:
            if "tensors sharing storage" in str(e):
                self.current_shard = {
                    k: v.clone() for k, v in self.current_shard.items()
                }
                safetensors.torch.save_file(
                    self.current_shard, shard_path, metadata={"format": "pt"}
                )
            else:
                raise e
