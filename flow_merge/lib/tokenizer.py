import logging
from copy import deepcopy
from itertools import combinations
from typing import Dict, Optional, Tuple, List
from pydantic import BaseModel, ConfigDict
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from flow_merge.lib import config
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.merge_plan import MergePlan

ADDITIONAL_SPECIAL_TOKENS_KEY = "additional_special_tokens"

logger = logging.getLogger(__name__)


class Tokenizer(BaseModel):
    tokenizer: PreTrainedTokenizerBase
    input_ids_mappings: Optional[Dict[str, Dict[int, int]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# Snapshot-less tokenizer implementation

class TokenizerValidator:
    @staticmethod
    def check_tokenizers_for_differences(tokenizers: Dict[str, PreTrainedTokenizerBase]) -> bool:
        differences_found = False

        for (model_a, tokenizer_a), (model_b, tokenizer_b) in combinations(tokenizers.items(), 2):
            differences_found |= TokenizerValidator._compare_tokenizer_vocabs(model_a, tokenizer_a, model_b,
                                                                              tokenizer_b)
            differences_found |= TokenizerValidator._compare_special_tokens(model_a, tokenizer_a, model_b, tokenizer_b)
            differences_found |= TokenizerValidator._compare_added_tokens_encoders(model_a, tokenizer_a, model_b,
                                                                                   tokenizer_b)

        return differences_found

    @staticmethod
    def _compare_tokenizer_vocabs(
            model_a: str,
            tokenizer_a: PreTrainedTokenizerBase,
            model_b: str,
            tokenizer_b: PreTrainedTokenizerBase,
    ) -> bool:
        vocab_a = tokenizer_a.get_vocab()
        vocab_b = tokenizer_b.get_vocab()

        if vocab_a != vocab_b:
            logger.info(f"Tokenizer for model {model_a} has different vocab compared to model {model_b}.")
            return True
        return False

    @staticmethod
    def _compare_special_tokens(
            model_a: str,
            tokenizer_a: PreTrainedTokenizerBase,
            model_b: str,
            tokenizer_b: PreTrainedTokenizerBase,
    ) -> bool:
        special_tokens_a = tokenizer_a.special_tokens_map
        special_tokens_b = tokenizer_b.special_tokens_map

        if special_tokens_a != special_tokens_b:
            logger.info(f"Tokenizer for model {model_a} has different special tokens compared to model {model_b}.")
            return True
        return False

    @staticmethod
    def _compare_added_tokens_encoders(
            model_a: str,
            tokenizer_a: PreTrainedTokenizerBase,
            model_b: str,
            tokenizer_b: PreTrainedTokenizerBase,
    ) -> bool:
        added_tokens_encoder_a = tokenizer_a.added_tokens_encoder
        added_tokens_encoder_b = tokenizer_b.added_tokens_encoder

        if added_tokens_encoder_a != added_tokens_encoder_b:
            logger.info(
                f"Tokenizer for model {model_a} has different added tokens encoder compared to model {model_b}.")
            return True
        return False


class TokenizerMerger:
    def __init__(
            self,
            base_model: str,
            tokenizers: Dict[str, PreTrainedTokenizerBase],
    ):
        self.base_model = base_model
        self.tokenizers = tokenizers

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
            token = added_token
            if token in merged_added_tokens:
                if merged_added_tokens[token] != added_token and token not in duplicate_added_tokens:
                    logger.warning(
                        f"Token {token} added with multiple different settings, using the first one by default.")
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
                logger.info(f"Adding additional special tokens: {special_token}.")
                merged_tokenizer.add_special_tokens({ADDITIONAL_SPECIAL_TOKENS_KEY: special_token})
            else:
                logger.warning(
                    f"Overriding {special_token_type} with {special_token}. When a conflict occurs, the last one takes priority.")
                merged_tokenizer.add_special_tokens({special_token_type: special_token})

        return merged_tokenizer


class MergeTokenizerService:
    def __init__(self, app_config: ApplicationConfig):
        self.config = app_config

    def get_merge_tokenizer(self, merge_plan: MergePlan) -> Tokenizer:
        all_models = list(
            set([source.model for slice in merge_plan.slices for source in slice.sources] + [merge_plan.base_model])
        )
        all_tokenizers = self._load_all_tokenizers(all_models)

        if not TokenizerValidator.check_tokenizers_for_differences(all_tokenizers):
            logger.info(
                f"No differences in tokens or vocab among tokenizers. Using {merge_plan.base_model} for the tokenizer.")
            return Tokenizer(tokenizer=all_tokenizers[merge_plan.base_model])

        logger.info("Different tokens or vocab among tokenizers. Building the tokenizer for the merged model.")

        merge_tokenizer = self.construct_appropriate_tokenizer(merge_plan.tokenizer_mode, merge_plan.base_model,
                                                               all_tokenizers)
        input_ids_mappings = self._create_input_ids_mappings(
            all_models,
            all_tokenizers,
            merge_tokenizer,
        )

        return Tokenizer(tokenizer=merge_tokenizer, input_ids_mappings=input_ids_mappings)

    @staticmethod
    def construct_appropriate_tokenizer(
            tokenizer_mode: str, base_model: str,
            all_tokenizers: Dict[str, PreTrainedTokenizerBase]
    ) -> PreTrainedTokenizerBase:
        if tokenizer_mode == "base":
            return all_tokenizers[base_model]

        builder = TokenizerMerger(
            base_model=base_model,
            tokenizers=all_tokenizers,
        )
        return builder.construct_merged_tokenizer()

    def _load_all_tokenizers(self, models_ids: List[str]) -> Dict[str, PreTrainedTokenizerBase]:

        all_tokenizers = {}
        for model_id in models_ids:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=self.config.trust_remote_code,
                )
            except Exception as e:
                error_message = f"Error loading tokenizer for {model_id}: {e}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            all_tokenizers[model_id] = tokenizer
        return all_tokenizers

    def _create_input_ids_mappings(
            self,
            models: List[str],
            all_tokenizers: Dict[str, PreTrainedTokenizerBase],
            merge_tokenizer: PreTrainedTokenizerBase,
    ) -> Dict[str, Dict[int, int]]:
        logger.info("Creating input ids mappings for interpolation of `embed_tokens` and `lm_head` layers.")
        input_ids_mappings = {}
        merge_tokenizer_vocab = merge_tokenizer.get_vocab()

        for model in models:
            vocab = all_tokenizers[model].get_vocab()
            vocab_size = self._get_model_vocab_size(model=model) or len(vocab)

            model_input_ids_mappings = {}
            for token, new_input_id in merge_tokenizer_vocab.items():
                old_input_id = vocab.get(token, -1)
                if old_input_id >= vocab_size:
                    raise RuntimeError(
                        f"{model} token {token} has input id {old_input_id} > {vocab_size - 1} due to trimming or modification.")
                model_input_ids_mappings[new_input_id] = old_input_id

            assert len(merge_tokenizer_vocab) == len(
                model_input_ids_mappings), "Lengths of merge_tokenizer_vocab and model_input_ids_mappings must be equal."

            input_ids_mappings[model] = model_input_ids_mappings

        return input_ids_mappings

    def _get_model_vocab_size(self, model: str) -> Optional[int]:
        try:
            model_config = AutoConfig.from_pretrained(
                self.config.local_dir / model,
                trust_remote_code=self.config.trust_remote_code
            )
            return model_config.vocab_size
        except Exception as e:
            logger.warning(f"Can't get vocab size for {model}: {e}")
            return None
