# test_model.py
import unittest
from unittest.mock import patch, MagicMock
from flow_merge.lib.model import Model
from flow_merge.lib.merge_settings import DirectorySettings


class TestModel(unittest.TestCase):
    @patch("flow_merge.lib.model_metadata.ModelMetadataService.load_model_info")
    @patch("flow_merge.lib.model_metadata.AutoConfig.from_pretrained")
    def test_from_path(self, mock_from_pretrained, mock_load_model_info):
        mock_metadata = MagicMock()
        mock_load_model_info.return_value = mock_metadata
        mock_config = MagicMock()
        mock_from_pretrained.return_value = mock_config

        directory_settings = DirectorySettings(local_dir=Path("./models"))
        model = Model.from_path("path/to/model", "fake-token", directory_settings)

        self.assertEqual(model.path, "path/to/model")
        self.assertEqual(model.metadata, mock_metadata)
        self.assertEqual(
            model.config, mock_config if mock_metadata.has_config else None
        )
        mock_load_model_info.assert_called_once_with("path/to/model")
        if mock_metadata.has_config:
            mock_from_pretrained.assert_called_once_with("path/to/model")

    @patch("flow_merge.lib.model_metadata.ModelMetadataService.load_model_info")
    def test_from_path_no_config(self, mock_load_model_info):
        mock_metadata = MagicMock()
        mock_metadata.has_config = False
        mock_load_model_info.return_value = mock_metadata

        directory_settings = DirectorySettings(local_dir=Path("./models"))
        model = Model.from_path("path/to/model", "fake-token", directory_settings)

        self.assertEqual(model.path, "path/to/model")
        self.assertEqual(model.metadata, mock_metadata)
        self.assertIsNone(model.config)
        mock_load_model_info.assert_called_once_with("path/to/model")