import unittest

from unittest.mock import patch
from pathlib import Path

from flow_merge.lib.model.model import Model, ModelId
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.logger import Logger
from flow_merge.lib.validators import DirectorySettings

class TestModel(unittest.TestCase):

    @patch('flow_merge.lib.model.model.ApplicationConfig')
    @patch('flow_merge.lib.model.model.ModelService')
    @patch('flow_merge.lib.model.model.TensorIndexService')
    @patch('flow_merge.lib.model.model.ModelMetadataService')
    def test_from_path(self, MockModelMetadataService, MockTensorIndexService, MockModelService, MockApplicationConfig):
        # Mock the return values for dependencies
        mock_metadata_service = MockModelMetadataService.return_value
        mock_metadata_service.load_model_info.return_value = ModelMetadata(
            id='test-model', sha='12345', file_list=['file1', 'file2'], has_adapter=False, has_safetensor_files=False, has_pytorch_bin_index=False, hf_exists=False, directory_settings=DirectorySettings(local_dir=Path('/tmp')))
        
        mock_tensor_index_service = MockTensorIndexService.create_file_to_tensor_index
        mock_tensor_index_service.return_value = {'file1': ['tensor1', 'tensor2']}
        
        mock_model_service = MockModelService.create_shard_files
        mock_model_service.return_value = []

        # Mock the ApplicationConfig
        mock_env = MockApplicationConfig.return_value
        mock_env.device = 'cpu'
        mock_env.hf_token = 'hf_validtoken123'

        # Define the input parameters
        path = Path('/models/test-model')
        directory_settings = DirectorySettings(local_dir=Path('/tmp'))
        logger = Logger()

        # Call the method
        model = Model.from_path(path, directory_settings, mock_env, logger)

        # Assertions
        self.assertIsInstance(model, Model)
        self.assertEqual(model.id, ModelId('/models/test-model'))
        self.assertEqual(model.path, path.resolve())
        self.assertIsInstance(model.metadata, ModelMetadata)
        self.assertEqual(model.file_to_tensor_index, {'file1': ['tensor1', 'tensor2']})
        self.assertEqual(model.shards, [])
        self.assertFalse(model.is_partial)

    @patch('flow_merge.lib.model.model.ApplicationConfig')
    @patch('flow_merge.lib.model.model.ModelService')
    @patch('flow_merge.lib.model.model.TensorIndexService')
    @patch('flow_merge.lib.model.model.ModelMetadataService')
    def test_from_layers(self, MockModelMetadataService, MockTensorIndexService, MockModelService, MockApplicationConfig):
        # Mock the return values for dependencies
        mock_metadata_service = MockModelMetadataService.return_value
        mock_metadata_service.load_model_info.return_value = ModelMetadata(
            id='test-model', sha='12345', file_list=['file1', 'file2'], has_adapter=True, has_safetensor_files=False, has_pytorch_bin_index=True, hf_exists=True, directory_settings=DirectorySettings(local_dir=Path('/tmp')))
        
        mock_tensor_index_service = MockTensorIndexService.create_file_to_tensor_index
        mock_tensor_index_service.return_value = {'file1': ['tensor1', 'tensor2']}
        
        mock_model_service = MockModelService.create_shard_files
        mock_model_service.return_value = []

        # Mock the ApplicationConfig
        mock_env = MockApplicationConfig.return_value
        mock_env.device = 'cpu'
        mock_env.hf_token = 'hf_validtoken123'

        # Define the input parameters
        layers_to_download = ['model.norm.weight', 'model.layers.0.mlp.gate_proj.weight']
        path = Path('/models/test-model')
        directory_settings = DirectorySettings(local_dir=Path('/tmp'))
        logger = Logger()

        # Call the method
        model = Model.from_layers(layers_to_download, path, directory_settings, mock_env, logger)

        # Assertions
        self.assertIsInstance(model, Model)
        self.assertEqual(model.id, ModelId('/models/test-model'))
        self.assertEqual(model.path, path.resolve())
        self.assertIsInstance(model.metadata, ModelMetadata)
        self.assertEqual(model.file_to_tensor_index, {'file1': ['tensor1', 'tensor2']})
        self.assertEqual(model.shards, [])
        self.assertTrue(model.is_partial)  # Since metadata has adapter and tensor index is None

if __name__ == '__main__':
    unittest.main()
