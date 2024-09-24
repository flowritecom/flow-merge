import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.config import ApplicationConfig, DeviceIdentifier
from flow_merge.lib.model.metadata import ModelMetadata, ModelMetadataService
from flow_merge.lib.tensor.index import TensorIndexService
from flow_merge.lib.tensor.loader import ShardFile, TensorRepository
from flow_merge.lib.tensor.writer import TensorWriter
from flow_merge.lib.file_io import FileRepository

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelService:
    """Manages the overall process of handling models."""

    def __init__(self, tensor_repository: TensorRepository, tensor_index_service: TensorIndexService,
                 file_repository: FileRepository, metadata_service: ModelMetadataService, config: ApplicationConfig):
        self.file_repository = file_repository
        self.tensor_repository = tensor_repository
        self.tensor_index_service = tensor_index_service
        self.metadata_service = metadata_service
        self.config = config

    def download_and_return_shard_file(
            self,
            output_dir: Path,
            repo_id: str,
            device: DeviceIdentifier,
            shard_filename: str,
            keys: Optional[List[str]] = None,
    ) -> ShardFile:
        output_path = output_dir / shard_filename
        self.file_repository.download_file(repo_id=repo_id, filename=shard_filename, download_dir=output_dir)

        if keys is None:
            try:
                keys = self.tensor_repository.get_tensor_keys_from_file(output_path, device)
            except RuntimeError as e:
                logger.warning(f"Tensor keys cannot be retrieved: {e}", e)
                keys = []  # Default to an empty list if keys cannot be retrieved

        return ShardFile(filename=shard_filename, path=output_path, tensor_keys=keys)

    def download_and_return_shard_files(
            self,
            file_to_tensor_index: Dict[str, List[str]],
            output_dir: Path,
            repo_id: str,
            device: DeviceIdentifier,
    ) -> List[ShardFile]:
        try:
            return [
                self.download_and_return_shard_file(
                    output_dir, repo_id, device, filename, keys
                )
                for filename, keys in file_to_tensor_index.items()
                if filename.endswith((".safetensors", ".bin"))
            ]
        except Exception as e:
            raise RuntimeError(f"Error gathering shard files: {e}")

    def gather_shard_files_from_layers(
            self,
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
                self.download_and_return_shard_file(
                    output_model_path, repo_id, device, filename
                )
                for filename in shards_to_download
            ]
        except Exception as e:
            raise RuntimeError(
                f"Error gathering shard files from layers {e}"
            )

    def create_shard_files(
            self,
            model_metadata: ModelMetadata, layers_to_download: List[str] = None,
    ) -> List[ShardFile]:
        if not model_metadata.has_config and not model_metadata.has_tokenizer_config:
            raise FileNotFoundError("Model is missing config.json or tokenizer_config.json")

        output_model_path = model_metadata.absolute_path

        # Download *minimal* required files to fetch information about shards
        self.file_repository.download_required_files(model_metadata)

        # If it has adapter, we continue with different procedure (how different though?)
        if model_metadata.has_adapter:
            return self.merge_and_save_model(model_metadata)

        # Multiple shards
        file_index = self.tensor_index_service.create_file_to_tensor_index(model_metadata)
        if file_index:
            if layers_to_download:
                self.gather_shard_files_from_layers(
                    layers_to_download,
                    file_index,
                    output_model_path,
                    model_metadata.id,
                    self.config.device
                )

            file_index = self.tensor_index_service.flip_keys(file_index)

            return self.download_and_return_shard_files(
                file_index, output_model_path, model_metadata.id, self.config.device
            )

        # Single-shard-file model
        logger.info("Index files not found, using single shard file fallback.")
        single_file = (
            "model.safetensors" if model_metadata.has_safetensor_files else "pytorch_model.bin"
        )
        shard_file = self.download_and_return_shard_file(
            output_model_path, model_metadata.id, self.config.device, single_file
        )
        return [shard_file]

    def save_model_shards(
            base_model: torch.nn.Module, output_dir: Path
    ) -> List[Path]:
        logger.info("Saving model shard files: save_model_shards")
        shard_files = []
        with TensorWriter(output_dir) as writer:
            for name, param in base_model.named_parameters():
                shard_name = writer.save_tensor(weight=ModelWeight(name=name), tensor=param)
                # name is incorrect here
                shard_files.append(output_dir / shard_name)
            writer.finish()
        return shard_files

    def merge_and_save_model(self, model_metadata: ModelMetadata) -> List[ShardFile]:
        logger.info("Merging adapter and saving model: merge_and_save_model")
        self.file_repository.download_adapter_files(model_metadata)


        # Peft configuration of the adapter repo
        adapter_config = PeftConfig.from_pretrained(str(model_metadata.relative_path))

        # Download minimal required files of the base model
        base_model_metadata = self.metadata_service.load_model_metadata(adapter_config.base_model_name_or_path)
        self.file_repository.download_required_files(base_model_metadata)
        # Determine base_model shards
        files = base_model_metadata.file_list
        if base_model_metadata.has_safetensor_files:
            base_model_shards = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")] or [
                "model.safetensors"]
        else:
            base_model_shards = [f for f in files if f.startswith("pytorch_model-") and f.endswith(".bin")] or [
                "pytorch_model.bin"]

        # Model already merged
        if Path(model_metadata.relative_path / "config.json").exists():
            return [
                ShardFile(
                    filename=f,
                    path=model_metadata.absolute_path / f,
                    tensor_keys=self.tensor_repository.get_tensor_keys_from_file(model_metadata.absolute_path / f,
                                                                                 device=self.config.device)
                ) for f in base_model_shards]

        # Merge base model with adapter and get resulting model
        logger.info("Loading and applying adapters: load_and_apply_adapters")
        for shard_file in base_model_shards:
            self.file_repository.download_file(base_model_metadata.id, shard_file, base_model_metadata.absolute_path)

        base_model = AutoModelForCausalLM.from_pretrained(base_model_metadata.absolute_path)
        model = PeftModel.from_pretrained(base_model, peft_config=adapter_config, model_id=model_metadata.id)
        model.set_adapter("default")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(model_metadata.absolute_path)

        shard_files = [
            ShardFile(
                filename=f,
                path=model_metadata.absolute_path / f,
                tensor_keys=self.tensor_repository.get_tensor_keys_from_file(model_metadata.absolute_path / f,
                                                                             device=self.config.device)
            ) for f in base_model_shards]

        return shard_files
