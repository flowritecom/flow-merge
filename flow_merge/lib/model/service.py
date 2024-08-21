import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from peft import PeftConfig, PeftModel
from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.config import ApplicationConfig, DeviceIdentifier
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.tensor.index import TensorIndexService
from flow_merge.lib.tensor.loader import ShardFile, TensorRepository
from flow_merge.lib.tensor.writer import TensorWriter
from flow_merge.lib.file_io import FileRepository

logger = logging.getLogger(__name__)


class ModelService:
    """Manages the overall process of handling models."""

    @staticmethod
    def download_and_return_shard_file(
            output_dir: Path,
            repo_id: str,
            device: DeviceIdentifier,
            shard_filename: str,
            keys: Optional[List[str]] = None,
    ) -> ShardFile:
        output_path = output_dir / shard_filename
        FileRepository.download_file(repo_id=repo_id, filename=shard_filename, download_dir=output_dir)

        if keys is None:
            try:
                keys = TensorRepository.get_tensor_keys_from_file(output_path, device)
            except RuntimeError as e:
                logger.warning(f"Tensor keys cannot be retrieved: {e}", e)
                keys = []  # Default to an empty list if keys cannot be retrieved

        return ShardFile(filename=shard_filename, path=output_path, tensor_keys=keys)

    @staticmethod
    def download_and_return_shard_files(
            file_to_tensor_index: Dict[str, List[str]],
            output_dir: Path,
            repo_id: str,
            device: DeviceIdentifier,
    ) -> List[ShardFile]:
        try:
            return [
                ModelService.download_and_return_shard_file(
                    output_dir, repo_id, device, filename, keys
                )
                for filename, keys in file_to_tensor_index.items()
                if filename.endswith((".safetensors", ".bin"))
            ]
        except Exception as e:
            raise RuntimeError(f"Error gathering shard files: {e}")

    @staticmethod
    def gather_shard_files_from_layers(
            layers_to_download,
            file_index,
            output_model_path,
            repo_id,
            device
    ) -> List[ShardFile]:
        shards_to_download = [file_index[layer] for layer in layers_to_download if layer in file_index]
        shards_to_download = list(set(shards_to_download))

        try:
            return [
                ModelService.download_and_return_shard_file(
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
            model_metadata: ModelMetadata, app_config: ApplicationConfig, layers_to_download: List[str] = None,
    ) -> List[ShardFile]:
        if not model_metadata.has_config and not model_metadata.has_tokenizer_config:
            raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")

        output_model_path = model_metadata.absolute_path

        # Download *minimal* required files to fetch information about shards
        FileRepository.download_required_files(model_metadata, app_config)

        # If it has adapter, we continue with different procedure (how different though?)
        # fixme: go through has_adatper path
        if model_metadata.has_adapter:
            return ModelService.merge_and_save_model(model_metadata, app_config)

        # Multiple shards
        file_index = TensorIndexService.create_file_to_tensor_index(model_metadata)
        if file_index:
            if layers_to_download:
                ModelService.gather_shard_files_from_layers(
                    layers_to_download,
                    file_index,
                    output_model_path,
                    model_metadata.id,
                    app_config.device
                )

            file_index = TensorIndexService.flip_keys(file_index)

            return ModelService.download_and_return_shard_files(
                file_index, output_model_path, model_metadata.id, app_config.device
            )

        # Single-shard-file model
        logger.info("Index files not found, using single shard file fallback.")
        single_file = (
            "model.safetensors" if model_metadata.has_safetensor_files else "pytorch_model.bin"
        )
        shard_file = ModelService.download_and_return_shard_file(
            output_model_path, model_metadata.id, app_config.device, single_file
        )
        return [shard_file]

    # Merging adapters
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
        logger.info("Loading and applying adapters: load_and_apply_adapters")
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
        logger.info("Saving model shard files: save_model_shards")
        shard_files = []
        with TensorWriter(output_dir) as writer:
            for name, param in base_model.named_parameters():
                writer.save_tensor(ModelWeight(name=name), param)
                shard_files.append(ShardFile(filename=name, path=str(output_dir)))
            writer.finish()
        return shard_files

    @staticmethod
    def merge_and_save_model(
            model_metadata: ModelMetadata, env: ApplicationConfig
    ) -> List[ShardFile]:
        logger.info("Merging adapter and saving model: merge_and_save_model")
        FileRepository.download_adapter_files(model_metadata, env)
        base_model_shards = ModelService.determine_base_model_shards(model_metadata)
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]

        base_model = ModelService.load_and_apply_adapters(
            adapter_files=adapter_files,
            base_model_shards=base_model_shards,
            device=env.device,
            repo_id=model_metadata.id,
            local_dir=model_metadata.directory_settings.local_dir,
        )
        return ModelService.save_model_shards(
            base_model, model_metadata.directory_settings.output_dir
        )
