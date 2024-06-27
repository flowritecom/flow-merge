from pathlib import Path
from typing import Dict, Optional

from flow_merge.lib.model.metadata import FileRepository, ModelMetadata


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
        
        return shardfile_index["weight_map"]
