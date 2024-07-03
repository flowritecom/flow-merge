import unittest

from unittest.mock import Mock
from flow_merge.lib.model.metadata.file_metadata import FileMetadata
from flow_merge.lib.model.metadata.file_validators import FileListValidator
from flow_merge.lib.model.metadata import ModelMetadata
from flow_merge.lib.model.metadata.file_validators._integrity import has_config_json
from flow_merge.lib.model.metadata.file_validators._safetensors import has_safetensors_files, has_safetensors_index
from flow_merge.lib.model.metadata.file_validators._tokenizer import has_tokenizer_config, has_tokenizer_file
from flow_merge.lib.model.metadata.file_validators._pytorch_bin import has_pytorch_bin_files, has_pytorch_bin_index
from flow_merge.lib.model.metadata.file_validators._adapter import has_adapter_files

class TestFileValidators(unittest.TestCase):

    def setUp(self):
        self.file_list = [
            "vocab.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors", "merges.txt", "generation_config.json",
            "config.json", "README.md", "LICENSE", ".gitattributes"
        ]
        self.sharded_safetensors = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json"
        ]
        self.sharded_pytorch_bin = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            "pytorch_model.bin.index.json"
        ]
        self.adapter_files = [
            "adapter_1.bin", "adapter_2.safetensors"
        ]
    
    def test_has_config_json(self):
        self.assertTrue(has_config_json(self.file_list))
        self.assertFalse(has_config_json(["README.md", "LICENSE"]))

    def test_has_safetensors_files(self):
        self.assertTrue(has_safetensors_files(self.file_list))
        self.assertFalse(has_safetensors_files(["README.md", "LICENSE"]))
        self.assertTrue(has_safetensors_files(self.sharded_safetensors))
    
    def test_has_safetensors_index(self):
        self.assertFalse(has_safetensors_index(self.file_list))
        self.assertTrue(has_safetensors_index(self.sharded_safetensors))

    def test_has_tokenizer_config(self):
        self.assertTrue(has_tokenizer_config(self.file_list))
        self.assertFalse(has_tokenizer_config(["README.md", "LICENSE"]))

    def test_has_tokenizer_file(self):
        self.assertTrue(has_tokenizer_file(self.file_list))
        self.assertFalse(has_tokenizer_file(["README.md", "LICENSE"]))
    
    def test_has_pytorch_bin_files(self):
        self.assertFalse(has_pytorch_bin_files(self.file_list))
        self.assertTrue(has_pytorch_bin_files(self.sharded_pytorch_bin))

    def test_has_pytorch_bin_index(self):
        self.assertFalse(has_pytorch_bin_index(self.file_list))
        self.assertTrue(has_pytorch_bin_index(self.sharded_pytorch_bin))

    def test_has_adapter_files(self):
        self.assertFalse(has_adapter_files(self.file_list))
        self.assertTrue(has_adapter_files(self.adapter_files))


class TestFileListValidator(unittest.TestCase):

    def setUp(self):
        # Shared file list
        self.shared_file_list = [
            "vocab.json", "tokenizer_config.json", "tokenizer.json",
            "merges.txt", "generation_config.json", "config.json",
            "README.md", "LICENSE", ".gitattributes"
        ]
        
        # Safetensors single file
        self.safetensors_file_list = self.shared_file_list + [
            "model.safetensors"
        ]
        
        # PyTorch bin file list
        self.pytorch_bin_file_list = self.shared_file_list + [
            "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
            "pytorch_model.bin.index.json"
        ]

        # Safetensors with index file list
        self.safetensors_with_index_file_list = self.shared_file_list + [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model.safetensors.index.json"
        ]

        self.logger = Mock()

    def test_check_metadata_updates(self):
        file_metadata_list = [FileMetadata(filename=file) for file in self.safetensors_file_list]
        metadata = ModelMetadata(
            id="test_model", 
            sha="test_sha",
            config={"key": "value"},
            file_metadata_list=file_metadata_list
        )
        validator = FileListValidator(env=None, logger=self.logger)
        updated_metadata = validator.check(metadata)

        self.assertTrue(updated_metadata.has_config)
        self.assertTrue(updated_metadata.has_tokenizer_config)
        self.assertTrue(updated_metadata.has_vocab)
        self.assertTrue(updated_metadata.has_safetensor_files)
        self.assertFalse(updated_metadata.has_safetensors_index)
        self.assertFalse(updated_metadata.has_pytorch_bin_files)
        self.assertFalse(updated_metadata.has_pytorch_bin_index)
        self.assertFalse(updated_metadata.has_adapter)

    def test_missing_files_logging(self):
        incomplete_file_list = [
            "README.md", "LICENSE", "vocab.json"
        ]
        incomplete_file_metadata_list = [FileMetadata(filename=file) for file in incomplete_file_list]
        incomplete_metadata = ModelMetadata(
            id="test_model", 
            sha="test_sha",
            config={"key": "value"},
            file_metadata_list=incomplete_file_metadata_list
        )
        validator = FileListValidator(env=None, logger=self.logger)
        validator.check(incomplete_metadata)
        
        self.logger.info.assert_any_call("Missing config.json file")
        self.logger.info.assert_any_call("Missing tokenizer_config.json file")
        self.logger.info.assert_any_call("Missing tokenizer vocabulary file")
        self.logger.info.assert_any_call("Missing .safetensors files")
        self.logger.info.assert_any_call("Missing model.safetensors.index.json file")
        self.logger.info.assert_any_call("Missing pytorch_model .bin files")
        self.logger.info.assert_any_call("Missing pytorch_model.bin.index.json file")
        self.logger.info.assert_any_call("Missing adapter files")

    def test_pytorch_bin_files_without_safetensors(self):
        pytorch_file_metadata_list = [FileMetadata(filename=file) for file in self.pytorch_bin_file_list]
        pytorch_metadata = ModelMetadata(
            id="test_model", 
            sha="test_sha",
            config={"key": "value"},
            file_metadata_list=pytorch_file_metadata_list
        )
        validator = FileListValidator(env=None, logger=self.logger)
        updated_metadata = validator.check(pytorch_metadata)

        self.assertTrue(updated_metadata.has_config)
        self.assertTrue(updated_metadata.has_tokenizer_config)
        self.assertTrue(updated_metadata.has_vocab)
        self.assertFalse(updated_metadata.has_safetensor_files)
        self.assertFalse(updated_metadata.has_safetensors_index)
        self.assertTrue(updated_metadata.has_pytorch_bin_files)
        self.assertTrue(updated_metadata.has_pytorch_bin_index)
        self.assertFalse(updated_metadata.has_adapter)

    def test_safetensors_with_index(self):
        safetensors_file_metadata_list = [FileMetadata(filename=file) for file in self.safetensors_with_index_file_list]
        safetensors_metadata = ModelMetadata(
            id="test_model", 
            sha="test_sha",
            config={"key": "value"},
            file_metadata_list=safetensors_file_metadata_list
        )
        validator = FileListValidator(env=None, logger=self.logger)
        updated_metadata = validator.check(safetensors_metadata)

        self.assertTrue(updated_metadata.has_config)
        self.assertTrue(updated_metadata.has_tokenizer_config)
        self.assertTrue(updated_metadata.has_vocab)
        self.assertTrue(updated_metadata.has_safetensor_files)
        self.assertTrue(updated_metadata.has_safetensors_index)
        self.assertFalse(updated_metadata.has_pytorch_bin_files)
        self.assertFalse(updated_metadata.has_pytorch_bin_index)
        self.assertFalse(updated_metadata.has_adapter)

if __name__ == "__main__":
    unittest.main()
