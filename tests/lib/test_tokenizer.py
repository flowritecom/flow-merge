import unittest
from unittest.mock import MagicMock, patch
from transformers import PreTrainedTokenizerBase
from flow_merge.lib.tokenizer import TokenizerLoader, TokenizerValidator, TokenizerMerger, InputIDsMapper, MergeTokenizerService
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger
from flow_merge.lib.model import Model


class TestTokenizerLoader(unittest.TestCase):
    @patch('flow_merge.lib.tokenizer.AutoTokenizer.from_pretrained')
    def test_load_all_tokenizers(self, mock_from_pretrained):
        # Mock objects
        mock_enriched_snapshot = MagicMock(EnrichedSnapshot)
        mock_env = MagicMock(ApplicationConfig)
        mock_logger = MagicMock(Logger)
        mock_logger.error = MagicMock()  # Add error method to logger mock
        mock_model = MagicMock(Model)
        
        # Add required attributes to mocks
        mock_model.path = "path/to/model"
        mock_enriched_snapshot.models = [mock_model]
        mock_enriched_snapshot.base_model = mock_model
        mock_enriched_snapshot.settings = MagicMock()  # Add settings attribute
        mock_enriched_snapshot.settings.hf_hub_settings.trust_remote_code = True
        mock_from_pretrained.return_value = MagicMock(PreTrainedTokenizerBase)
        
        # Call the method
        tokenizers = TokenizerLoader.load_all_tokenizers(mock_enriched_snapshot, mock_env, mock_logger)
        
        # Assertions
        self.assertIn(mock_model, tokenizers)
        mock_from_pretrained.assert_called_with("path/to/model", trust_remote_code=True)


class TestTokenizerValidator(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(Logger)
        self.mock_logger.info = MagicMock()  # Add info method to logger mock
        self.mock_model_a = MagicMock(Model)
        self.mock_model_b = MagicMock(Model)
        self.mock_tokenizer_a = MagicMock(PreTrainedTokenizerBase)
        self.mock_tokenizer_b = MagicMock(PreTrainedTokenizerBase)

    def test_check_tokenizers_for_differences(self):
        tokenizers = {
            self.mock_model_a: self.mock_tokenizer_a,
            self.mock_model_b: self.mock_tokenizer_b
        }
        
        # Mock method returns
        self.mock_tokenizer_a.get_vocab.return_value = {"token_a": 0}
        self.mock_tokenizer_b.get_vocab.return_value = {"token_b": 1}
        self.mock_tokenizer_a.special_tokens_map = {"special_token_a": "<s_a>"}
        self.mock_tokenizer_b.special_tokens_map = {"special_token_b": "<s_b>"}
        self.mock_tokenizer_a.added_tokens_encoder = {"added_token_a": 2}
        self.mock_tokenizer_b.added_tokens_encoder = {"added_token_b": 3}

        differences_found = TokenizerValidator.check_tokenizers_for_differences(tokenizers, self.mock_logger)
        
        self.assertTrue(differences_found)


class TestTokenizerMerger(unittest.TestCase):
    def setUp(self):
        self.mock_base_model = MagicMock(Model)
        self.mock_tokenizers = {
            self.mock_base_model: MagicMock(PreTrainedTokenizerBase)
        }
        self.mock_env = MagicMock(ApplicationConfig)
        self.mock_logger = MagicMock(Logger)
        self.mock_logger.warning = MagicMock()  # Add warning method to logger mock
        self.mock_logger.info = MagicMock()  # Add info method to logger mock
        self.merger = TokenizerMerger(self.mock_base_model, self.mock_tokenizers, self.mock_env, self.mock_logger)

    @patch('flow_merge.lib.tokenizer.deepcopy')
    def test_construct_merged_tokenizer(self, mock_deepcopy):
        mock_tokenizer = MagicMock(PreTrainedTokenizerBase)
        mock_deepcopy.return_value = mock_tokenizer
        
        merged_tokenizer = self.merger.construct_merged_tokenizer()
        
        self.assertIsInstance(merged_tokenizer, PreTrainedTokenizerBase)
        mock_deepcopy.assert_called()


class TestInputIDsMapper(unittest.TestCase):
    @patch('flow_merge.lib.tokenizer.AutoConfig.from_pretrained')
    def test_create_input_ids_mappings(self, mock_from_pretrained):
        # Mock objects
        mock_enriched_snapshot = MagicMock(EnrichedSnapshot)
        mock_logger = MagicMock(Logger)
        mock_logger.info = MagicMock()  # Add info method to logger mock
        mock_logger.warning = MagicMock()  # Add warning method to logger mock
        mock_model = MagicMock(Model)
        mock_tokenizer = MagicMock(PreTrainedTokenizerBase)
        
        # Add required attributes to mocks
        mock_enriched_snapshot.models = [mock_model]
        mock_enriched_snapshot.base_model = mock_model
        mock_enriched_snapshot.hf_hub_settings = MagicMock()  # Add hf_hub_settings attribute
        mock_enriched_snapshot.hf_hub_settings.trust_remote_code = True
        mock_tokenizer.get_vocab.return_value = {"token": 0}
        mock_from_pretrained.return_value.vocab_size = 1

        all_tokenizers = {mock_model: mock_tokenizer}
        merge_tokenizer = MagicMock(PreTrainedTokenizerBase)
        merge_tokenizer.get_vocab.return_value = {"token": 0}
        
        mappings = InputIDsMapper.create_input_ids_mappings(mock_enriched_snapshot, all_tokenizers, merge_tokenizer, mock_logger)
        
        self.assertIn(mock_model, mappings)
        self.assertIn(0, mappings[mock_model])


class TestMergeTokenizerService(unittest.TestCase):
    @patch('flow_merge.lib.tokenizer.TokenizerLoader.load_all_tokenizers')
    @patch('flow_merge.lib.tokenizer.TokenizerValidator.check_tokenizers_for_differences')
    @patch('flow_merge.lib.tokenizer.InputIDsMapper.create_input_ids_mappings')
    def test_get_merge_tokenizer(self, mock_create_input_ids_mappings, mock_check_tokenizers_for_differences, mock_load_all_tokenizers):
        # Mock objects
        mock_enriched_snapshot = MagicMock(EnrichedSnapshot)
        mock_env = MagicMock(ApplicationConfig)
        mock_logger = MagicMock(Logger)
        mock_logger.info = MagicMock()  # Add info method to logger mock
        mock_model = MagicMock(Model)
        mock_tokenizer = MagicMock(PreTrainedTokenizerBase)
        
        # Add required attributes to mocks
        mock_enriched_snapshot.base_model = mock_model
        mock_enriched_snapshot.base_model.path = "path/to/base_model"
        mock_load_all_tokenizers.return_value = {mock_model: mock_tokenizer}
        mock_check_tokenizers_for_differences.return_value = False
        
        service = MergeTokenizerService(mock_env, mock_logger)
        tokenizer = service.get_merge_tokenizer(mock_enriched_snapshot)
        
        self.assertIsInstance(tokenizer.tokenizer, PreTrainedTokenizerBase)
        mock_load_all_tokenizers.assert_called()
        mock_check_tokenizers_for_differences.assert_called()
        mock_create_input_ids_mappings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
