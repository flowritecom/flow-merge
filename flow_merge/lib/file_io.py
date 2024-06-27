import json

from pathlib import Path
from huggingface_hub import hf_hub_download

from flow_merge.lib.model.metadata import ModelMetadata

class FileRepository:
    """Immutable repository for handling file operations."""

    @staticmethod
    def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
        try:
            file_path = hf_hub_download(
                repo_id,
                filename,
                local_dir=str(local_dir),  # Convert Path to str for hf_hub_download
                resume_download=True,
                token=config.hf_token,
            )
            return Path(file_path)  # Convert returned file_path to Path

        except FileExistsError:
            print(
                f"File {local_dir / filename} already exists and is complete, skipping download."
            )
            return local_dir / filename
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while downloading {filename} from {repo_id}: {e}"
            )

    @staticmethod
    def load_index(file_path: Path) -> dict:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error loading index from {file_path}: {e}")

    @staticmethod
    def download_required_files(metadata: "ModelMetadata"):
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer.vocab",
            "vocab.json",
            "tokenizer_config.json",
        ]
        for filename in required_files:
            if filename in metadata.file_list:
                FileRepository.download_file(
                    metadata.id, filename, metadata.directory_settings.local_dir
                )

    @staticmethod
    def download_adapter_files(model_metadata: "ModelMetadata"):
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]
        for adapter_file in adapter_files:
            FileRepository.download_file(
                model_metadata.id,
                adapter_file,
                model_metadata.directory_settings.local_dir,
            )