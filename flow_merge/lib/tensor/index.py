from pathlib import Path
from typing import Dict, Optional

from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.file_io import FileRepository


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
        safetensors_index_path = Path(metadata.absolute_path) / "model.safetensors.index.json"
        pytorch_bin_index_path = Path(metadata.absolute_path) / "pytorch_model.bin.index.json"

        if metadata.hf_exists and metadata.has_safetensors_index:
            try:
                index_path = FileRepository.download_file(
                    repo_id=str(metadata.id),
                    filename="model.safetensors.index.json",
                    download_dir=metadata.absolute_path,
                )
            except Exception as e:
                print(f"Safetensors index not found: {e}")
                return None
        elif metadata.hf_exists and metadata.has_pytorch_bin_index:
            try:
                index_path = FileRepository.download_file(
                    repo_id=str(metadata.id),
                    filename="pytorch_model.bin.index.json",
                    download_dir=metadata.absolute_path,
                )
            except Exception as e:
                print(f"Pytorch bin index not found: {e}")
                return None

        if index_path is None and safetensors_index_path.exists():
            index_path = safetensors_index_path
        elif index_path is None and pytorch_bin_index_path.exists():
            index_path = pytorch_bin_index_path
        else:
            return None

        shardfile_index = FileRepository.load_model_files_index(index_path)

        return shardfile_index["weight_map"]
