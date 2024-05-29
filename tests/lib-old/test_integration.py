from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from flow_merge.lib.merge_settings import DirectorySettings
from flow_merge.lib.model import Model


@patch("flow_merge.lib.model.Model.config", new_callable=lambda: Any)
@patch("transformers.PretrainedConfig")
@patch("flow_merge.lib.model_metadata.huggingface_hub.hf_api.repo_info")
@patch("flow_merge.lib.model_metadata.huggingface_hub.hf_hub_download")
def test_model_loading(
    mock_hf_hub_download, mock_repo_info, mock_pretrained_config, mock_config
):
    mock_repo_info.return_value = MagicMock()
    mock_hf_hub_download.return_value = "local_path/to/file"
    mock_pretrained_config.return_value = MagicMock()

    # code merge required to function, ref: 1f7e1f0 that adds directory_settings to ModelMetadata
    # https://github.com/flowritecom/flow-merge/commit/1f7e1f0b712bb21d8685ec7bfec05d2dea258475
    directory_settings = DirectorySettings(local_dir="./models")
    model = Model.from_path("path/to/model", "fake-token", directory_settings)

    assert model.metadata is not None
    assert model.path.startswith("path/to/model")
