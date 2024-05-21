import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import huggingface_hub
from pydantic import BaseModel, Field
from transformers import AutoConfig, PretrainedConfig
from huggingface_hub.hf_api import (
    ModelInfo,
    BlobLfsInfo,
    SafeTensorsInfo,
    ModelCardData,
    TransformersInfo,
    RepoSibling,
)

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
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_settings import DirectorySettings
from flow_merge.lib.config import config

logger = get_logger(__name__)

CHUNK_SIZE = 64 * 1024  # 64KB


class FileMetadata(BaseModel):
    filename: str
    sha: Optional[str] = None
    blob_id: Optional[str] = None
    size: Optional[int] = None
    lfs: Optional[BlobLfsInfo] = None


class ModelMetadata(BaseModel):
    id: str
    sha: Optional[str]
    config: Optional[Dict]
    file_metadata_list: List[FileMetadata] = Field(default_factory=list)
    file_list: List[str] = Field(default_factory=list)
    safetensors_info: Optional[SafeTensorsInfo] = Field(
        alias="safetensors", default=None
    )

    relative_path: Optional[Path] = None
    absolute_path: Optional[Path] = None
    directory_settings: Optional[DirectorySettings] = None

    hf_author: Optional[str] = Field(alias="author", default=None)
    hf_created_at: Optional[datetime] = Field(alias="created_at", default=None)
    hf_last_modified: Optional[datetime] = Field(alias="last_modified", default=None)
    hf_private: bool = Field(alias="private", default=None)
    hf_gated: Optional[Literal["auto", "manual", False]] = Field(
        alias="gated", default=None
    )
    hf_disabled: Optional[bool] = Field(alias="disabled", default=None)
    hf_library_name: Optional[str] = Field(alias="library_name", default=None)
    hf_tags: List[str] = Field(alias="tags", default_factory=list)
    hf_pipeline_tag: Optional[str] = Field(alias="pipeline_tag", default=None)
    hf_mask_token: Optional[str] = Field(alias="mask_token", default=None)
    hf_card_data: Optional[ModelCardData] = Field(alias="card_data", default=None)
    hf_widget_data: Optional[Any] = Field(alias="widget_data", default=None)
    hf_model_index: Optional[Dict] = Field(alias="model_index", default=None)
    hf_transformers_info: Optional[TransformersInfo] = Field(
        alias="transformers_info", default=None
    )
    hf_data: Optional[ModelInfo] = Field(alias="data", default=None)

    hf_exists: bool = True
    has_config: bool = False
    has_vocab: bool = False
    has_tokenizer_config: bool = False
    has_pytorch_bin_index: bool = False
    has_safetensors_index: bool = False
    has_safetensor_files: bool = False
    has_pytorch_bin_files: bool = False
    has_adapter: bool = False

    def update_checks(self):
        if self.file_metadata_list:
            self.file_list = [
                file_metadata.filename for file_metadata in self.file_metadata_list
            ]
            self.has_config = has_config_json(self.file_list)
            self.has_vocab = has_tokenizer_file(self.file_list)
            self.has_tokenizer_config = has_tokenizer_config(self.file_list)
            self.has_pytorch_bin_index = has_pytorch_bin_index(self.file_list)
            self.has_safetensors_index = has_safetensors_index(self.file_list)
            self.has_safetensor_files = has_safetensors_files(self.file_list)
            self.has_pytorch_bin_files = has_pytorch_bin_files(self.file_list)
            self.has_adapter = has_adapter_files(self.file_list)


class ModelMetadataService:
    def __init__(self, directory_settings: DirectorySettings = DirectorySettings()):
        self.directory_settings = directory_settings

    @staticmethod
    def generate_content_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def download_hf_file(self, repo_id: str, filename: str) -> str:
        return huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            local_dir=self.directory_settings.local_dir,
            resume_download=True,
            token=config.hf_token,
        )

    def fetch_hf_model_info(self, repo_id: str) -> ModelInfo:
        return huggingface_hub.hf_api.repo_info(
            repo_id=repo_id,
            repo_type="model",
            files_metadata=True,
            token=config.hf_token,
        )

    def create_file_metadata_list_from_hf(
        self, hf_model_info: ModelInfo, repo_id: str
    ) -> List[FileMetadata]:
        def create_file_metadata(sibling: RepoSibling) -> FileMetadata:
            if sibling.lfs is None:
                path_to_downloaded_file = self.download_hf_file(
                    repo_id, sibling.rfilename
                )
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
            for file_path in path_to_model.glob("*")
        ]

    def load_model_info(self, path_or_id: str) -> ModelMetadata:
        path = Path(path_or_id)
        try:
            hf_model_info = self.fetch_hf_model_info(path_or_id)
            file_metadata_list = self.create_file_metadata_list_from_hf(
                hf_model_info, path_or_id
            )

            model_metadata = ModelMetadata(
                **hf_model_info.__dict__,
                file_metadata_list=file_metadata_list,
                relative_path=path,
                absolute_path=path.resolve(),
            )
            model_metadata.update_checks()

            return model_metadata
        except huggingface_hub.hf_api.RepositoryNotFoundError:
            logger.info(
                "Model not found in Hugging face. Inferring from local model directory."
            )
            path_to_model = (self.directory_settings.local_dir / path_or_id).resolve()
            if path_to_model.exists():
                file_metadata_list = self.create_file_metadata_list_from_local(
                    path_to_model
                )
                config = None
                try:
                    config_obj = PretrainedConfig.from_json_file(
                        str(path_to_model / "config.json")
                    )
                    config = config_obj.to_dict()
                except EnvironmentError as e:
                    logger.warn(f"Error while fetching config for local model: {e}")

                model_metadata = ModelMetadata(
                    id=path_or_id,
                    sha=None,
                    file_metadata_list=file_metadata_list,
                    config=config,
                    hf_exists=False,
                    relative_path=path_to_model,
                    absolute_path=path_to_model.resolve(),
                    directory_settings=self.directory_settings,
                )
                model_metadata.update_checks()
                return model_metadata
            else:
                logger.warn("Model not found locally, cannot create model metadata.")
                return ModelMetadata(id=path_or_id, hf_exists=False)
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return ModelMetadata(id=path_or_id, hf_exists=False)
