import json
import logging

from pathlib import Path
from huggingface_hub import hf_hub_download

from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.config import ApplicationConfig


# Fixme: HuggingFace download client instead of FileRepository?
class FileRepository:
    """Immutable repository for handling file operations."""

    @staticmethod
    def download_file(repo_id: str, filename: str, download_dir: Path,
                      env: ApplicationConfig = ApplicationConfig()) -> Path:
        try:
            if Path(download_dir / filename).exists():
                print(f"File {download_dir / filename} already exists, skipping download")
                return Path(download_dir / filename)

            # FIXME: local_dir arg should be called download_dir so we know it shouldn't be modified after given as arg
            file_path = hf_hub_download(
                repo_id,
                filename,
                local_dir=str(download_dir),  # Convert Path to str for hf_hub_download
                token=env.hf_token,
            )
            print(f"downloaded file {repo_id}/{filename} to {file_path}")
            return Path(file_path)  # Convert returned file_path to Path

        except FileExistsError:
            print(
                f"File {download_dir / filename} already exists, skipping download."
            )
            return download_dir / filename
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while downloading {filename} from {repo_id}: {e}"
            )

    @staticmethod
    def load_model_files_index(file_path: Path) -> dict:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error loading index from {file_path}: {e}")

    @staticmethod
    def download_required_files(metadata: ModelMetadata, app_config: ApplicationConfig):
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer.vocab",
            "vocab.json",
            "tokenizer_config.json",
        ]
        for filename in required_files:
            if filename not in metadata.file_list:
                continue
            print(f"Downloading required file {filename} into {str(metadata.relative_path)}")
            FileRepository.download_file(metadata.id, filename, metadata.relative_path, app_config)

    @staticmethod
    def download_adapter_files(model_metadata: "ModelMetadata", env: ApplicationConfig):
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]
        for adapter_file in adapter_files:
            print(
                f"Downloading adapter file {adapter_file} into {str(model_metadata.directory_settings.local_dir / model_metadata.id)}")
            FileRepository.download_file(
                model_metadata.id,
                adapter_file,
                model_metadata.directory_settings.local_dir / model_metadata.id,
                env
            )
