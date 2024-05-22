import hashlib
import pytest
from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from typing import Optional
from flow_merge.lib.merge_settings import DirectorySettings
from flow_merge.lib.config import CentralConfig
from huggingface_hub.hf_api import ModelInfo, RepoSibling, RepositoryNotFoundError
from flow_merge.lib.model_metadata import (
    ModelMetadataService,
    FileMetadata,
    ModelMetadata,
)

##### Utilities
def create_model_info() -> ModelInfo:
    sibling = RepoSibling(
        rfilename="dummy_file", size=1234, lfs=None
    )

    # ModelInfo expects a list of dict for siblings and creates a RepoSibling list.
    # Internally, it incorrectly attempts to get attribute blobId instead of blob_id
    # Hence, I've had to resort to None for blob_id and pass the RepoSibling as a dict.

    return ModelInfo(
        id="dummy_id",
        siblings=[sibling.__dict__],
        private=False,
        downloads=0,
        likes=0,
        tags=["dummy_tag"],
    )


def create_directory_settings() -> DirectorySettings:
    return DirectorySettings(local_dir="./models")


def create_model_metadata_service(
        directory_settings: Optional[DirectorySettings] = create_directory_settings()
    ) -> ModelMetadataService:
    return ModelMetadataService(
        directory_settings=directory_settings
    )
#####

# High Priority Tests
def test_generate_content_hash():
    service = create_model_metadata_service()

    # Mock file content
    file_content = b"Hello, World!"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    with patch("builtins.open", mock_open(read_data=file_content)):
        actual_hash = service.generate_content_hash("dummy_path")

    assert (
        actual_hash == expected_hash
    ), "The generated content hash should match the expected hash"


@patch("huggingface_hub.hf_hub_download")
@patch("flow_merge.lib.model_metadata.config")
def test_download_hf_file(mock_config, mock_download):

    directory_settings = create_directory_settings()

    service = ModelMetadataService(
        directory_settings=directory_settings
    )

    mock_config.hf_token = "dummy_token"
    mock_download.return_value = "path/to/downloaded/file"

    result = service.download_hf_file(
        repo_id="dummy_repo",
        filename="dummy_file"
    )

    assert result == "path/to/downloaded/file"
    mock_download.assert_called_once_with(
        "dummy_repo",
        "dummy_file",
        local_dir=directory_settings.local_dir,
        resume_download=True,
        token="dummy_token",
    )


@patch("huggingface_hub.hf_api.repo_info")
def test_fetch_hf_model_info(mock_repo_info):
    service = ModelMetadataService(
        token="dummy_token", base_path=Path(".")
    )
    mock_repo_info.return_value = ModelInfo(
        id="dummy_id",
        private=False,
        downloads=0,
        likes=0,
        tags=["dummy_tag"],
    )

    result = service.fetch_hf_model_info("dummy_repo")

    assert result.id == "dummy_id"
    mock_repo_info.assert_called_once_with(
        repo_id="dummy_repo",
        repo_type="model",
        files_metadata=True,
        token="dummy_token",
    )


# Medium Priority Tests
@patch.object(ModelMetadataService, "download_hf_file")
@patch.object(ModelMetadataService, "generate_content_hash")
def test_create_file_metadata_list_from_hf(
    mock_generate_content_hash, mock_download_hf_file
):
    service = ModelMetadataService(
        token="dummy_token", base_path=Path(".")
    )
    mock_generate_content_hash.return_value = "dummy_sha"
    mock_download_hf_file.return_value = "path/to/downloaded/file"

    model_info = create_model_info()

    result = service.create_file_metadata_list_from_hf(
        hf_model_info=model_info, 
        repo_id="dummy_repo",
        model_path="dummy_model_path"
    )

    expected = [
        FileMetadata(
            filename="dummy_file",
            sha="dummy_sha",
            size=1234,
            lfs=None,
        )
    ]

    assert result == expected
    mock_download_hf_file.assert_called_once_with("dummy_repo", "dummy_model_path", "dummy_file")
    mock_generate_content_hash.assert_called_once_with("path/to/downloaded/file")


@patch("flow_merge.lib.model_metadata.ModelMetadataService.generate_content_hash")
@patch("pathlib.Path.glob")
def test_create_file_metadata_list_from_local(mock_glob, mock_generate_content_hash):
    service = ModelMetadataService(
        token="", base_path=Path(".")
    )
    mock_generate_content_hash.return_value = "dummy_sha"

    mock_file = MagicMock(spec=Path)
    mock_file.stat.return_value.st_size = 1234
    mock_file.name = "dummy_file"
    mock_glob.return_value = [mock_file]

    result = service.create_file_metadata_list_from_local(Path("dummy_path"))

    expected = [FileMetadata(filename="dummy_file", sha="dummy_sha", size=1234)]

    assert result == expected
    mock_generate_content_hash.assert_called_once_with(mock_file)


# Low Priority Tests
@patch.object(ModelMetadataService, "fetch_hf_model_info")
@patch.object(ModelMetadataService, "create_file_metadata_list_from_hf")
def test_load_model_info_hf(
    mock_create_file_metadata_list_from_hf, mock_fetch_hf_model_info
):
    service = ModelMetadataService(
        token="dummy_token", base_path="dummy_base_path"
    )

    model_info = create_model_info()

    mock_fetch_hf_model_info.return_value = model_info
    mock_create_file_metadata_list_from_hf.return_value = []

    result = service.load_model_info(path_or_id="dummy_repo")

    assert isinstance(result, ModelMetadata)
    assert result.id == "dummy_id"
    mock_fetch_hf_model_info.assert_called_once_with("dummy_repo")
    mock_create_file_metadata_list_from_hf.assert_called_once_with(
        model_info, "dummy_repo", "dummy_base_path"
    )


@patch.object(ModelMetadataService, "create_file_metadata_list_from_local")
@patch.object(ModelMetadataService, "fetch_hf_model_info")
@patch("transformers.AutoConfig.from_pretrained")
@patch("pathlib.Path.resolve")
def test_load_model_info_local(
    mock_path_resolve, mock_from_pretrained, mock_fetch_hf_model_info, mock_create_file_metadata_list_from_local
):
    service = ModelMetadataService(
        token="",
        base_path="dummy_base_path"
    )

    mock_create_file_metadata_list_from_local.return_value = []
    mock_from_pretrained.return_value.to_dict.return_value = {
        "config_key": "config_value"
    }

    mock_fetch_hf_model_info.side_effect = RepositoryNotFoundError("Model not found")
    mock_path_resolve.return_value = Path("dummy_base_path/dummy_repo")

    with patch("pathlib.Path.exists", return_value=True):
        result = service.load_model_info(path_or_id="dummy_repo")

    assert isinstance(result, ModelMetadata)
    assert result.id == "dummy_repo"
    assert result.config == {"config_key": "config_value"}
    mock_create_file_metadata_list_from_local.assert_called_once_with(
        Path("dummy_base_path/dummy_repo")
    )
    mock_from_pretrained.assert_called_once_with(Path("dummy_base_path/dummy_repo"))


@patch.object(ModelMetadataService, "fetch_hf_model_info")
@patch.object(ModelMetadataService, "download_hf_file")
@patch.object(ModelMetadataService, "generate_content_hash")
def test_end_to_end_hf_model(
    mock_generate_content_hash, mock_download_hf_file, mock_fetch_hf_model_info
):
    service = ModelMetadataService(
        token="dummy_token", base_path="dummy_base_path"
    )

    model_info = create_model_info()
    
    mock_fetch_hf_model_info.return_value = model_info

    mock_generate_content_hash.return_value = "dummy_sha"
    mock_download_hf_file.return_value = "path/to/downloaded/file"

    result = service.load_model_info(path_or_id="dummy_repo")

    assert isinstance(result, ModelMetadata)
    assert result.id == "dummy_id"
    assert result.file_metadata_list[0].sha == "dummy_sha"
    mock_fetch_hf_model_info.assert_called_once_with("dummy_repo")
    mock_download_hf_file.assert_called_once_with("dummy_repo", "dummy_base_path", "dummy_file")
    mock_generate_content_hash.assert_called_once_with("path/to/downloaded/file")


@patch.object(ModelMetadataService, "create_file_metadata_list_from_local")
@patch("transformers.AutoConfig.from_pretrained")
@patch("pathlib.Path.resolve")
def test_end_to_end_local_model(
    mock_path_resolve, mock_from_pretrained, mock_create_file_metadata_list_from_local
):
    service = ModelMetadataService(
        token="",
        base_path="dummy_base_path"
    )

    mock_create_file_metadata_list_from_local.return_value = []
    mock_from_pretrained.return_value.to_dict.return_value = {
        "config_key": "config_value"
    }
    mock_path_resolve.return_value = Path("dummy_base_path/dummy_repo")

    with patch("pathlib.Path.exists", return_value=True):
        result = service.load_model_info(path_or_id="dummy_repo")

    assert isinstance(result, ModelMetadata)
    assert result.id == "dummy_repo"
    assert result.config == {"config_key": "config_value"}
    mock_create_file_metadata_list_from_local.assert_called_once_with(
        Path("dummy_base_path/dummy_repo")
    )
    mock_from_pretrained.assert_called_once_with(Path("dummy_base_path/dummy_repo"))
