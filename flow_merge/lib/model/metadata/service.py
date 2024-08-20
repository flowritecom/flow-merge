import hashlib
import logging
import huggingface_hub
from pathlib import Path
from typing import List
from huggingface_hub.hf_api import (
    ModelInfo,
    RepoSibling,
)
from transformers import PretrainedConfig
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.model.metadata import ModelMetadata

CHUNK_SIZE = 64 * 1024


class ModelMetadataService:
    def __init__(self, app_config: ApplicationConfig):
        self.app_config = app_config

    def load_model_metadata(self, id: str) -> ModelMetadata:
        path_to_model = (self.app_config.local_dir / id)
        if path_to_model.resolve().exists():
            try:
                logging.info("Model found locally, loading from local directory")
                return self._load_local_model_metadata(id, path_to_model)
            except EnvironmentError as e:
                logging.warning("Failed to load model info from local file, trying HuggingFace", e)

        try:
            return self._load_hf_model_metadata(id, path_to_model)
        except huggingface_hub.hf_api.RepositoryNotFoundError as e:
            raise Exception("Model not found in HuggingFace. Cannot load model information.") from e
        except Exception as e:
            raise Exception(f"Failed to load model info. {e}") from e

    def _load_hf_model_metadata(self, id: str, path_to_model: Path):
        hf_model_info = huggingface_hub.hf_api.repo_info(
            repo_id=id,
            repo_type="model",
            files_metadata=True,
            token=self.app_config.hf_token,
        )
        model_metadata = ModelMetadata(
            **hf_model_info.__dict__,
            relative_path=path_to_model,
            absolute_path=path_to_model.resolve(),
        )
        return model_metadata

    def _load_local_model_metadata(self, id: str, path_to_model: Path):

        all_files = [file_path.name for file_path in path_to_model.glob("*")]
        model_metadata = ModelMetadata(
            id=id,
            file_list=all_files,
            config=PretrainedConfig.from_json_file(str(path_to_model / "config.json")).to_dict(),
            hf_exists=False,
            relative_path=path_to_model,
            absolute_path=path_to_model.resolve(),
            has_config=Path(path_to_model / "config.json").exists(),
            has_vocab="tokenizer.json" in all_files or any(file.endswith("tokenizer.vocab") for file in all_files),
            has_tokenizer_config="tokenizer_config.json" in all_files,
            has_pytorch_bin_index=any(file.endswith(".bin.index.json") for file in all_files),
            has_safetensors_index=any(file.endswith(".safetensors.index.json") for file in all_files),
            has_safetensor_files=self._has_safetensors_files(all_files),
            has_pytorch_bin_files=self._has_pytorch_bin_files(all_files),
            has_adapter=any(
                file.startswith("adapter_")
                and (file.endswith(".bin") or file.endswith(".safetensors"))
                for file in all_files
            ),
        )
        return model_metadata

    ## FIXME: WHY THIS NOT USING FILEIO REPOSITORY !!?
    def download_hf_file(self, repo_id: str, filename: str) -> str:
        return huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            local_dir=self.app_config.local_dir / repo_id,
            resume_download=True,
            token=self.app_config.hf_token,
        )

    @staticmethod
    def _has_pytorch_bin_files(file_list: List[str]):
        pytorch_bin_files = [
            file for file in file_list
            if file.endswith(".bin") and not file.endswith(".index.json")
        ]
        num_shards = len(pytorch_bin_files)
        if not num_shards:
            return False

        if num_shards == 1 and "pytorch_model.bin" in file_list:
            return True

        return all(
            f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin" in file_list
            for i in range(1, num_shards + 1)
        )

    @staticmethod
    def _has_safetensors_files(file_list: List[str]):
        safetensors_files = [file for file in file_list if file.endswith(".safetensors")]
        print(file_list)
        num_shards = len(safetensors_files)
        if not num_shards:
            return False

        if num_shards == 1 and "model.safetensors" in file_list:
            return True

        return all(
            f"model-{i:05d}-of-{num_shards:05d}.safetensors" in file_list
            for i in range(1, num_shards + 1)
        )
