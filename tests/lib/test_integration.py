# test_integration.py
import unittest
from unittest.mock import patch, MagicMock
from flow_merge.lib.model import Model
from flow_merge.lib.merge_settings import DirectorySettings


class TestIntegration(unittest.TestCase):
    @patch("flow_merge.lib.model_metadata.huggingface_hub.hf_api.repo_info")
    @patch("flow_merge.lib.model_metadata.huggingface_hub.hf_hub_download")
    def test_model_loading(self, mock_hf_hub_download, mock_repo_info):
        mock_repo_info.return_value = MagicMock()
        mock_hf_hub_download.return_value = "local_path/to/file"

        directory_settings = DirectorySettings(local_dir=Path("./models"))
        model = Model.from_path("path/to/model", "fake-token", directory_settings)

        self.assertIsNotNone(model.metadata)
        self.assertTrue(model.path.startswith("path/to/model"))


if __name__ == "__main__":
    unittest.main()
