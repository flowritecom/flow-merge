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
from flow_merge.lib.model.metadata import FileMetadata
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.model.metadata.file_validators import FileListValidator

CHUNK_SIZE = 64 * 1024


class ModelMetadataService:
    def __init__(self, app_config: ApplicationConfig):
        self.app_config = app_config
        self.metadata_files_validator = FileListValidator()

    def load_model_metadata(self, id: str) -> ModelMetadata:
        path_to_model = (self.app_config.local_dir / id)
        if path_to_model.resolve().exists():
            try:
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
            file_metadata_list=self.create_file_metadata_list_from_hf(hf_model_info, id),
            relative_path=path_to_model,
            absolute_path=path_to_model.resolve(),
        )
        self.metadata_files_validator.check(metadata=model_metadata)
        return model_metadata

    def _load_local_model_metadata(self, id: str, path_to_model: Path):
        model_metadata = ModelMetadata(
            id=id,
            sha=None,
            file_list=[file_path.name for file_path in path_to_model.glob("*")],
            file_metadata_list=self.create_file_metadata_list_from_local(path_to_model),
            config=PretrainedConfig.from_json_file(str(path_to_model / "config.json")).to_dict(),
            hf_exists=False,
            relative_path=path_to_model,
            absolute_path=path_to_model.resolve(),
        )
        self.metadata_files_validator.check(metadata=model_metadata)
        return model_metadata

    def generate_content_hash(self, file_path: str) -> str:
        logging.info("Generating Content Hash: generate_content_hash")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    ## FIXME: WHY THIS NOT USING FILEIO REPOSITORY !!?
    def download_hf_file(self, repo_id: str, filename: str) -> str:
        return huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            local_dir=self.app_config.local_dir / repo_id,
            resume_download=True,
            token=self.app_config.hf_token,
        )

    def create_file_metadata_list_from_hf(
            self, hf_model_info: ModelInfo, repo_id: str
    ) -> List[FileMetadata]:
        def create_file_metadata(sibling: RepoSibling) -> FileMetadata:
            if sibling.lfs is None:
                path_to_downloaded_file = self.download_hf_file(repo_id, sibling.rfilename)
                sha = self.generate_content_hash(path_to_downloaded_file)
            else:
                sha = sibling.lfs.sha256
            return FileMetadata(
                sha=sha,
                blob_id=sibling.blob_id,
                filename=sibling.rfilename,
                size=sibling.size,
                lfs=sibling.lfs,
            )

        return [create_file_metadata(sibling) for sibling in hf_model_info.siblings]

    def create_file_metadata_list_from_local(
            self, path_to_model: Path
    ) -> List[FileMetadata]:
        return [
            FileMetadata(
                sha=self.generate_content_hash(str(file_path)),
                size=file_path.stat().st_size,
                filename=file_path.name,
            )
            for file_path in path_to_model.glob("*") if file_path.is_file()
        ]
