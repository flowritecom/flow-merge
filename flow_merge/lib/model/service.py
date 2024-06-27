from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftConfig, PeftModel

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.constants import DeviceIdentifier
from flow_merge.lib.model.metadata import FileRepository, ModelMetadata
from flow_merge.lib.tensor.index import TensorIndexService
from flow_merge.lib.tensor.loader import ShardFile, TensorRepository
from flow_merge.lib.tensor.writer import TensorWriter


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
                keys = TensorRepository.get_tensor_keys(output_path, file_type, device)
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
    def get_output_model_path(model_metadata: ModelMetadata):
        return (
            model_metadata.directory_settings.output_dir / model_metadata.id
        )
    
    @staticmethod
    def validate_config(model_metadata: ModelMetadata):
        if not (model_metadata.has_config and model_metadata.has_tokenizer_config):
            raise FileNotFoundError(
                "Model is missing config.json or tokenizer_config.json"
            )

    @staticmethod
    def get_shard_filenames_from_layers(
        layers_to_download: List[str], 
        file_index: Dict
    ) -> List[str]:
        shard_filenames = set()

        (
            shard_filenames.add(file_index[layer]) 
            for layer in layers_to_download 
            if layer in file_index
        )
        
        return list(shard_filenames)
        
    @staticmethod
    def gather_shard_files_from_layers(
        layers_to_download,
        file_index,
        output_model_path,
        repo_id,
        device
    ) -> List[ShardFile]:
        shards_to_download = ModelService.get_shard_filenames_from_layers(
            layers_to_download,
            file_index
        )

        try:
            return [
                ModelService.create_shard_file(
                    output_model_path, repo_id, device, filename
                )
                for filename in shards_to_download
            ]
        except Exception as e:
            raise RuntimeError(
                f"Error gathering shard files from layers {e}"
            )

    @staticmethod
    def create_shard_files(
        model_metadata: ModelMetadata, device: DeviceIdentifier, layers_to_download: List[str] = None
    ) -> List[ShardFile]:
        output_model_path = ModelService.get_output_model_path(model_metadata)

        ModelService.validate_config(model_metadata)

        FileRepository.download_required_files(model_metadata)

        if model_metadata.has_adapter:
            return ModelService.merge_and_save_model(model_metadata, device)

        file_index = TensorIndexService.create_file_to_tensor_index(model_metadata)
        if file_index:
            if layers_to_download:
                ModelService.gather_shard_files_from_layers(
                    layers_to_download,
                    file_index,
                    output_model_path,
                    model_metadata.id,
                    device
                )

            file_index = TensorIndexService.flip_keys(file_index)

            return ModelService.gather_shard_files(
                file_index, output_model_path, model_metadata.id, device
            )
        else:
            print("Index files not found, using single shard file fallback.")
            single_file = (
                "model.safetensors"
                if model_metadata.has_safetensor_files
                else "pytorch_model.bin"
            )
            shard_file = ModelService.create_shard_file(
                output_model_path, model_metadata.id, device, single_file
            )
            return [shard_file]
        
