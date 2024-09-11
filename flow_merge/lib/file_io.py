import json
import logging
from concurrent.futures import ThreadPoolExecutor, wait

from pathlib import Path
from huggingface_hub import hf_hub_download

from flow_merge.lib import config
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.model.metadata import ModelMetadata

logger = logging.getLogger(__name__)


# Fixme: HuggingFace download client instead of FileRepository?
class FileRepository:
    """Immutable repository for handling file operations."""

    def __init__(self, app_config: ApplicationConfig):
        self.app_config = app_config

    @staticmethod
    def download_file(repo_id: str, filename: str, download_dir: Path) -> Path:
        app_config = config.app_config.get()
        print(f"Downloading {repo_id} to {download_dir / filename}")
        logger.debug(f"Downloading {repo_id} to {download_dir / filename}")
        try:
            if Path(download_dir / filename).exists():
                logger.debug(f"File {download_dir / filename} already exists, skipping download")
                return Path(download_dir / filename)

            file_path = hf_hub_download(repo_id, filename, local_dir=str(download_dir), token=app_config.hf_token)
            logger.debug(f"Downloaded file {repo_id}/{filename} to {file_path}")
            return Path(file_path)

        except FileExistsError:
            logger.debug(f"File {download_dir / filename} already exists, skipping download.")
            return download_dir / filename
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while downloading {filename} from {repo_id}: {e}")

    @staticmethod
    def load_model_files_index(file_path: Path) -> dict:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Error loading index from {file_path}: {e}")

    @staticmethod
    def download_required_files(metadata: ModelMetadata):
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer.vocab",
            "vocab.json",
            "tokenizer_config.json",
        ]
        # todo: parallelize
        for filename in required_files:
            if filename not in metadata.file_list:
                continue
            logger.debug(f"Downloading required file {filename} into {str(metadata.relative_path)}")
            FileRepository.download_file(metadata.id, filename, metadata.relative_path)

    @staticmethod
    def download_adapter_files(model_metadata: ModelMetadata):
        adapter_files = [f for f in model_metadata.file_list if "adapter" in f]

        for adapter_file in adapter_files:
            FileRepository.download_file(model_metadata.id, adapter_file, model_metadata.absolute_path)
