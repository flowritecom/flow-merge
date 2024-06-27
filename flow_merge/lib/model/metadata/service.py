import hashlib
import huggingface_hub

from pathlib import Path
from typing import List
from huggingface_hub.hf_api import (
    ModelInfo,
    RepoSibling,
)
from transformers import PretrainedConfig

from flow_merge.lib.model.metadata import FileMetadata
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.validators import DirectorySettings
from flow_merge.lib.model.metadata.file_validators import FileListValidator
from flow_merge.lib.constants import CHUNK_SIZE


class ModelMetadataService:
    def __init__(self, env, logger, directory_settings: DirectorySettings):
        self.env = env
        self.logger = logger
        self.directory_settings = directory_settings
        self.metadata_files_validator = FileListValidator(env=env, logger=logger)

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
            token=self.env.hf_token,
        )

    def fetch_hf_model_info(self, repo_id: str) -> ModelInfo:
        return huggingface_hub.hf_api.repo_info(
            repo_id=repo_id,
            repo_type="model",
            files_metadata=True,
            token=self.env.hf_token,
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
            self.metadata_files_validator.check(metadata=model_metadata)
            # model_metadata.update_checks()

            return model_metadata
        except huggingface_hub.hf_api.RepositoryNotFoundError:
            self.logger.info(
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
                    self.logger.warn(f"Error while fetching config for local model: {e}")

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
                self.metadata_files_validator.check(metadata=model_metadata)
                # model_metadata.update_checks()
                return model_metadata
            else:
                self.logger.warn("Model not found locally, cannot create model metadata.")
                return ModelMetadata(id=path_or_id, hf_exists=False)
        except Exception as e:
            self.logger.error(f"Error fetching model info: {e}")
            return ModelMetadata(id=path_or_id, hf_exists=False)
