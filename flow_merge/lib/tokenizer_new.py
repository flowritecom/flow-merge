from copy import deepcopy
from itertools import combinations
from typing import Dict, Optional, Tuple

from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from flow_merge.lib.constants import ADDITIONAL_SPECIAL_TOKENS_KEY
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger
from flow_merge.lib.merge_config import MergeConfig
from flow_merge.lib.model import Model


class Tokenizer(BaseModel):
    tokenizer: PreTrainedTokenizerBase
    input_ids_mappings: Optional[Dict[Model, Dict[int, int]]] = None

    class Config:
        allow_mutation = False


class TokenizerLoader:
    @staticmethod
    def load_all_tokenizers(merge_config: MergeConfig, env: ApplicationConfig, logger: Logger) -> Dict[Model, PreTrainedTokenizerBase]:
        all_tokenizers = {}
        for model in merge_config.models + [merge_config.base_model]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model.path,
                    trust_remote_code=merge_config.hf_hub_settings.trust_remote_code,
                )
            except Exception as e:
                error_message = f"Error loading tokenizer for {model}: {e}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            all_tokenizers[model] = tokenizer
        return all_tokenizers


class TokenizerValidator:
    @staticmethod
    def check_tokenizers_for_differences(tokenizers: Dict[Model, PreTrainedTokenizerBase], logger: Logger) -> bool:
        differences_found = False

        for (model_a, tokenizer_a), (model_b, tokenizer_b) in combinations(tokenizers.items(), 2):
            differences_found |= TokenizerValidator._compare_tokenizer_vocabs(model_a, tokenizer_a, model_b, tokenizer_b, logger)
            differences_found |= TokenizerValidator._compare_special_tokens(model_a, tokenizer_a, model_b, tokenizer_b, logger)
            differences_found |= TokenizerValidator._compare_added_tokens_encoders(model_a, tokenizer_a, model_b, tokenizer_b, logger)

        return differences_found

    @staticmethod
    def _compare_tokenizer_vocabs(
        model_a: Model,
        tokenizer_a: PreTrainedTokenizerBase,
        model_b: Model,
        tokenizer_b: PreTrainedTokenizerBase,
        logger: Logger
    ) -> bool:
        vocab_a = tokenizer_a.get_vocab()
        vocab_b = tokenizer_b.get_vocab()

        if vocab_a != vocab_b:
            logger.info(f"Tokenizer for model {model_a} has different vocab compared to model {model_b}.")
            return True
        return False

    @staticmethod
    def _compare_special_tokens(
        model_a: Model,
        tokenizer_a: PreTrainedTokenizerBase,
        model_b: Model,
        tokenizer_b: PreTrainedTokenizerBase,
        logger: Logger
    ) -> bool:
        special_tokens_a = tokenizer_a.special_tokens_map
        special_tokens_b = tokenizer_b.special_tokens_map

        if special_tokens_a != special_tokens_b:
            logger.info(f"Tokenizer for model {model_a} has different special tokens compared to model {model_b}.")
            return True
        return False

    @staticmethod
    def _compare_added_tokens_encoders(
        model_a: Model,
        tokenizer_a: PreTrainedTokenizerBase,
        model_b: Model,
        tokenizer_b: PreTrainedTokenizerBase,
        logger: Logger
    ) -> bool:
        added_tokens_encoder_a = tokenizer_a.added_tokens_encoder
        added_tokens_encoder_b = tokenizer_b.added_tokens_encoder

        if added_tokens_encoder_a != added_tokens_encoder_b:
            logger.info(f"Tokenizer for model {model_a} has different added tokens encoder compared to model {model_b}.")
            return True
        return False


class TokenizerMerger:
    def __init__(
            self, 
            base_model: Model, 
            tokenizers: Dict[Model, PreTrainedTokenizerBase],
            env: ApplicationConfig,
            logger: Logger
        ):
        self.base_model = base_model
        self.tokenizers = tokenizers
        self.env = env
        self.logger = logger

    def construct_merged_tokenizer(self) -> PreTrainedTokenizerBase:
        merged_vocab, merged_added_tokens, merged_special_tokens = self._merge_tokenizer_components()
        return self._create_merged_tokenizer(merged_vocab, merged_added_tokens, merged_special_tokens)

    def _merge_tokenizer_components(self) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
        merged_vocab = {}
        merged_added_tokens = {}
        merged_special_tokens = {}
        duplicate_added_tokens = set()

        for model, tokenizer in self.tokenizers.items():
            if model == self.base_model:
                continue
            vocab = tokenizer.get_vocab()
            added_tokens = tokenizer.added_tokens_decoder
            special_tokens = tokenizer.special_tokens_map

            self._merge_vocab(merged_vocab, vocab)
            self._merge_added_tokens(merged_added_tokens, added_tokens, duplicate_added_tokens)
            self._merge_special_tokens(merged_special_tokens, special_tokens)

        return merged_vocab, merged_added_tokens, merged_special_tokens

    def _merge_vocab(self, merged_vocab: Dict[str, int], vocab: Dict[str, int]) -> None:
        for token, input_id in vocab.items():
            if token not in merged_vocab:
                merged_vocab[token] = len(merged_vocab)

    def _merge_added_tokens(
        self,
        merged_added_tokens: Dict[str, str],
        added_tokens: Dict[int, str],
        duplicate_added_tokens: set
    ) -> None:
        for input_id, added_token in added_tokens.items():
            token = added_token.content
            if token in merged_added_tokens:
                if merged_added_tokens[token] != added_token and token not in duplicate_added_tokens:
                    self.logger.warning(f"Token {token} added with multiple different settings, using the first one by default.")
                    duplicate_added_tokens.add(token)
            else:
                merged_added_tokens[token] = added_token

    def _merge_special_tokens(
        self, merged_special_tokens: Dict[str, str], special_tokens: Dict[str, str]
    ) -> None:
        for special_token_type, special_token in special_tokens.items():
            if special_token_type == ADDITIONAL_SPECIAL_TOKENS_KEY and isinstance(special_token, list):
                merged_special_tokens.setdefault(special_token_type, []).extend(special_token)
            else:
                merged_special_tokens[special_token_type] = special_token

    def _create_merged_tokenizer(
        self,
        merged_vocab: Dict[str, int],
        merged_added_tokens: Dict[str, str],
        merged_special_tokens: Dict[str, str],
    ) -> PreTrainedTokenizerBase:
        base_tokenizer = self.tokenizers[self.base_model]
        merged_tokenizer = deepcopy(base_tokenizer)

        base_vocab_set = set(base_tokenizer.get_vocab())
        base_added_tokens_set = set(base_tokenizer.added_tokens_decoder.values())

        tokens_to_add = [token for token in merged_vocab if token not in base_vocab_set]
        merged_tokenizer.add_tokens(tokens_to_add)

        tokens_to_add_with_settings = [
            merged_added_tokens[token]
            for token in merged_added_tokens
            if token not in base_added_tokens_set
        ]
        merged_tokenizer.add_tokens(tokens_to_add_with_settings)

        for special_token_type, special_token in merged_special_tokens.items():
            if special_token_type == ADDITIONAL_SPECIAL_TOKENS_KEY and isinstance(special_token, list):
                self.logger.info(f"Adding additional special tokens: {special_token}.")
                merged_tokenizer.add_special_tokens({ADDITIONAL_SPECIAL_TOKENS_KEY: special_token})
            else:
                self.logger.warning(f"Overriding {special_token_type} with {special_token}. When a conflict occurs, the last one takes priority.")
                merged_tokenizer.add_special_tokens({special_token_type: special_token})

        return merged_tokenizer


class InputIDsMapper:
    @staticmethod
    def create_input_ids_mappings(
        merge_config: MergeConfig,
        all_tokenizers: Dict[Model, PreTrainedTokenizerBase],
        merge_tokenizer: PreTrainedTokenizerBase,
        logger: Logger
    ) -> Dict[Model, Dict[int, int]]:
        logger.info("Creating input ids mappings for interpolation of `embed_tokens` and `lm_head` layers.")
        input_ids_mappings = {}
        merge_tokenizer_vocab = merge_tokenizer.get_vocab()

        for model in merge_config.models + [merge_config.base_model]:
            vocab = all_tokenizers[model].get_vocab()
            vocab_size = InputIDsMapper.get_vocab_size(
                model=model,
                trust_remote_code=merge_config.hf_hub_settings.trust_remote_code,
                logger=logger
            ) or len(vocab)

            model_input_ids_mappings = {}
            for token, new_input_id in merge_tokenizer_vocab.items():
                old_input_id = vocab.get(token, -1)
                if old_input_id >= vocab_size:
                    raise RuntimeError(f"{model} token {token} has input id {old_input_id} > {vocab_size-1} due to trimming or modification.")
                model_input_ids_mappings[new_input_id] = old_input_id

            assert len(merge_tokenizer_vocab) == len(model_input_ids_mappings), "Lengths of merge_tokenizer_vocab and model_input_ids_mappings must be equal."

            input_ids_mappings[model] = model_input_ids_mappings

        return input_ids_mappings

    @staticmethod
    def get_vocab_size(model: Model, trust_remote_code: bool, logger: Logger) -> Optional[int]:
        try:
            model_config = AutoConfig.from_pretrained(model.path, trust_remote_code=trust_remote_code)
            return model_config.vocab_size
        except Exception as e:
            logger.warning(f"Can't get vocab size for {model}: {e}")
            return None


class MergeTokenizerService:

    def __init__(self, env: ApplicationConfig, logger: Logger):
        self.env = env
        self.logger = logger

    def get_merge_tokenizer(self, merge_config: MergeConfig) -> Tokenizer:
        all_tokenizers = TokenizerLoader.load_all_tokenizers(merge_config, self.env, self.logger)

        if not TokenizerValidator.check_tokenizers_for_differences(all_tokenizers, self.logger):
            self.logger.info(f"No differences in tokens or vocab among tokenizers. Using {merge_config.base_model.path} for the tokenizer.")
            return Tokenizer(tokenizer=all_tokenizers[merge_config.base_model])

        self.logger.info("Different tokens or vocab among tokenizers. Building the tokenizer for the merged model.")

        merge_tokenizer = self.construct_appropriate_tokenizer(merge_config, all_tokenizers)
        input_ids_mappings = InputIDsMapper.create_input_ids_mappings(
            merge_config, 
            all_tokenizers, 
            merge_tokenizer, 
            self.logger
        )

        return Tokenizer(tokenizer=merge_tokenizer, input_ids_mappings=input_ids_mappings)

    def construct_appropriate_tokenizer(
        self, merge_config: MergeConfig, all_tokenizers: Dict[Model, PreTrainedTokenizerBase]
    ) -> PreTrainedTokenizerBase:
        if merge_config.tokenizer_settings.mode == "base":
            return all_tokenizers[merge_config.base_model]

        builder = TokenizerMerger(
            base_model=merge_config.base_model, 
            tokenizers=all_tokenizers,
            env=self.env,
            logger=self.logger
            )
        return builder.construct_merged_tokenizer()