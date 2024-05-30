from copy import deepcopy
from itertools import combinations
from typing import Dict, Optional, Tuple

from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from flow_merge.lib.constants import ADDITIONAL_SPECIAL_TOKENS_KEY
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import MergeConfig, Model
from flow_merge.lib.model import Model

logger = get_logger(__name__)


class Tokenizer(BaseModel):
    tokenizer: PreTrainedTokenizerBase
    input_ids_mappings: Optional[Dict[Model, Dict[int, int]]] = None

    class Config:
        allow_mutation = False


def get_vocab_size(model: Model, trust_remote_code: bool) -> Optional[int]:
    """
    Get tokenizer vocabulary size from model config.

    Args:
        model: The model to get the vocabulary size from.
        trust_remote_code: Whether to trust remote code.

    Returns:
        The vocabulary size of the tokenizer, or None if it cannot be obtained.
    """
    try:
        model_config = AutoConfig.from_pretrained(
            model.path, trust_remote_code=trust_remote_code
        )
        return model_config.vocab_size
    except Exception as e:
        logger.warning(f"Can't get vocab size for {model}: {e}")
        return None


def check_tokenizers_for_differences(
    tokenizers: Dict[Model, PreTrainedTokenizerBase],
) -> bool:
    """
    Check if any tokenizer has different tokens, vocab, or added tokens compared to other tokenizers.

    Args:
        tokenizers: A dictionary of tokenizers with Model's as keys.

    Returns:
        True if there are differences in tokens, vocab, or added tokens, False otherwise.
    """
    differences_found = False

    for (model_a, tokenizer_a), (model_b, tokenizer_b) in combinations(
        tokenizers.items(), 2
    ):
        differences_found |= compare_tokenizer_vocabs(
            model_a, tokenizer_a, model_b, tokenizer_b
        )
        differences_found |= compare_special_tokens(
            model_a, tokenizer_a, model_b, tokenizer_b
        )
        differences_found |= compare_added_tokens_encoders(
            model_a, tokenizer_a, model_b, tokenizer_b
        )

    return differences_found


def compare_tokenizer_vocabs(
    model_a: Model,
    tokenizer_a: PreTrainedTokenizerBase,
    model_b: Model,
    tokenizer_b: PreTrainedTokenizerBase,
) -> bool:
    """
    Compare vocabularies of two tokenizers.

    Args:
        model_a: The first model.
        tokenizer_a: The first tokenizer.
        model_b: The second model.
        tokenizer_b: The second tokenizer.

    Returns:
        True if vocabularies are different, False otherwise.
    """
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()

    if vocab_a != vocab_b:
        logger.info(
            f"Tokenizer for model {model_a} has different vocab compared to model {model_b}."
        )
        return True
    return False


def compare_special_tokens(
    model_a: Model,
    tokenizer_a: PreTrainedTokenizerBase,
    model_b: Model,
    tokenizer_b: PreTrainedTokenizerBase,
) -> bool:
    """
    Compare special tokens of two tokenizers.

    Args:
        model_a: The first model.
        tokenizer_a: The first tokenizer.
        model_b: The second model.
        tokenizer_b: The second tokenizer.

    Returns:
        True if special tokens are different, False otherwise.
    """
    special_tokens_a = tokenizer_a.special_tokens_map
    special_tokens_b = tokenizer_b.special_tokens_map

    if special_tokens_a != special_tokens_b:
        logger.info(
            f"Tokenizer for model {model_a} has different special tokens compared to model {model_b}."
        )
        return True
    return False


def compare_added_tokens_encoders(
    model_a: Model,
    tokenizer_a: PreTrainedTokenizerBase,
    model_b: Model,
    tokenizer_b: PreTrainedTokenizerBase,
) -> bool:
    """
    Compare added tokens encoders of two tokenizers.

    Args:
        model_a: The first model.
        tokenizer_a: The first tokenizer.
        model_b: The second model.
        tokenizer_b: The second tokenizer.

    Returns:
        True if added tokens encoders are different, False otherwise.
    """
    added_tokens_encoder_a = tokenizer_a.added_tokens_encoder
    added_tokens_encoder_b = tokenizer_b.added_tokens_encoder

    if added_tokens_encoder_a != added_tokens_encoder_b:
        logger.info(
            f"Tokenizer for model {model_a} has different added tokens encoder compared to model {model_b}."
        )
        return True
    return False


class MergedTokenizerBuilder(BaseModel):
    base_model: Model
    tokenizers: Dict[Model, PreTrainedTokenizerBase]

    def construct_merged_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Constructs a tokenizer with a vocab that contains all tokens from all models involved in the merge.

        Returns:
            The merged tokenizer.
        """
        (
            merged_vocab,
            merged_added_tokens,
            merged_special_tokens,
        ) = self._merge_tokenizer_components()
        return self._create_merged_tokenizer(
            merged_vocab, merged_added_tokens, merged_special_tokens
        )

    def _merge_tokenizer_components(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
        """
        Merge components of tokenizers including vocab, added tokens, and special tokens.

        Returns:
            A tuple of merged vocab, merged added tokens, and merged special tokens.
        """
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
            self._merge_added_tokens(
                merged_added_tokens, added_tokens, duplicate_added_tokens
            )
            self._merge_special_tokens(merged_special_tokens, special_tokens)

        return merged_vocab, merged_added_tokens, merged_special_tokens

    def _merge_vocab(self, merged_vocab: Dict[str, int], vocab: Dict[str, int]) -> None:
        """
        Merge the vocabulary of a tokenizer into the merged vocabulary.

        Args:
            merged_vocab: The merged vocabulary.
            vocab: The vocabulary of a tokenizer.
        """
        for token, input_id in vocab.items():
            if token not in merged_vocab:
                merged_vocab[token] = len(merged_vocab)

    def _merge_added_tokens(
        self,
        merged_added_tokens: Dict[str, str],
        added_tokens: Dict[int, str],
        duplicate_added_tokens: set,
    ) -> None:
        """
        Merge the added tokens of a tokenizer into the merged added tokens.

        Args:
            merged_added_tokens: The merged added tokens.
            added_tokens: The added tokens of a tokenizer.
            duplicate_added_tokens: A set to track duplicate added tokens.
        """
        for input_id, added_token in added_tokens.items():
            token = added_token.content
            if token in merged_added_tokens:
                if (
                    merged_added_tokens[token] != added_token
                    and token not in duplicate_added_tokens
                ):
                    logger.warning(
                        f"Token {token} added with multiple different settings, using the first one by default."
                    )
                    duplicate_added_tokens.add(token)
            else:
                merged_added_tokens[token] = added_token

    def _merge_special_tokens(
        self, merged_special_tokens: Dict[str, str], special_tokens: Dict[str, str]
    ) -> None:
        """
        Merge the special tokens of a tokenizer into the merged special tokens.

        Args:
            merged_special_tokens: The merged special tokens.
            special_tokens: The special tokens of a tokenizer.
        """
        for special_token_type, special_token in special_tokens.items():
            if special_token_type == ADDITIONAL_SPECIAL_TOKENS_KEY and isinstance(
                special_token, list
            ):
                merged_special_tokens.setdefault(special_token_type, []).extend(
                    special_token
                )
            else:
                merged_special_tokens[special_token_type] = special_token

    def _create_merged_tokenizer(
        self,
        merged_vocab: Dict[str, int],
        merged_added_tokens: Dict[str, str],
        merged_special_tokens: Dict[str, str],
    ) -> PreTrainedTokenizerBase:
        """
        Create the merged tokenizer using the merged components.

        Args:
            merged_vocab: The merged vocabulary.
            merged_added_tokens: The merged added tokens.
            merged_special_tokens: The merged special tokens.

        Returns:
            The merged tokenizer.
        """
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
            if special_token_type == ADDITIONAL_SPECIAL_TOKENS_KEY and isinstance(
                special_token, list
            ):
                logger.info(f"Adding additional special tokens: {special_token}.")
                merged_tokenizer.add_special_tokens(
                    {ADDITIONAL_SPECIAL_TOKENS_KEY: special_token}
                )
            else:
                logger.warning(
                    f"Overriding {special_token_type} with {special_token}. When a conflict occurs, the last one takes priority."
                )
                merged_tokenizer.add_special_tokens({special_token_type: special_token})

        return merged_tokenizer


def get_merge_tokenizer(merge_config: MergeConfig) -> Tokenizer:
    """
    Returns a tokenizer for the merged model based on the provided configuration.

    Args:
        merge_config: The configuration object containing the merge configuration.

    Returns:
        The tokenizer for the merged model with the inputs_ids_mappings if available.
    """
    all_tokenizers = load_all_tokenizers(merge_config)

    if not check_tokenizers_for_differences(all_tokenizers):
        logger.info(
            f"No differences in tokens or vocab among tokenizers. Using {merge_config.base_model.path} for the tokenizer."
        )
        return Tokenizer(tokenizer=all_tokenizers[merge_config.base_model])

    logger.info(
        "Different tokens or vocab among tokenizers. Building the tokenizer for the merged model."
    )
    merge_tokenizer = construct_appropriate_tokenizer(merge_config, all_tokenizers)
    input_ids_mappings = create_input_ids_mappings(
        merge_config, all_tokenizers, merge_tokenizer
    )

    return Tokenizer(tokenizer=merge_tokenizer, input_ids_mappings=input_ids_mappings)


def load_all_tokenizers(
    merge_config: MergeConfig,
) -> Dict[Model, PreTrainedTokenizerBase]:
    """
    Load all tokenizers for the given merge configuration.

    Args:
        merge_config: The configuration object containing the merge configuration.

    Returns:
        A dictionary of all loaded tokenizers.
    """
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


def construct_appropriate_tokenizer(
    merge_config: MergeConfig, all_tokenizers: Dict[Model, PreTrainedTokenizerBase]
) -> PreTrainedTokenizerBase:
    """
    Construct the appropriate tokenizer based on the merge configuration.

    Args:
        merge_config: The configuration object containing the merge configuration.
        all_tokenizers: A dictionary of all loaded tokenizers.

    Returns:
        The appropriate tokenizer for the merged model.
    """
    if merge_config.tokenizer_settings.mode == "base":
        return all_tokenizers[merge_config.base_model]

    builder = MergedTokenizerBuilder(
        base_model=merge_config.base_model, tokenizers=all_tokenizers
    )
    return builder.construct_merged_tokenizer()


def create_input_ids_mappings(
    merge_config: MergeConfig,
    all_tokenizers: Dict[Model, PreTrainedTokenizerBase],
    merge_tokenizer: PreTrainedTokenizerBase,
) -> Dict[Model, Dict[int, int]]:
    """
    Create input IDs mappings for interpolation of `embed_tokens` and `lm_head` layers.

    Args:
        merge_config: The configuration object containing the merge configuration.
        all_tokenizers: A dictionary of all tokenizers.
        merge_tokenizer: The merged tokenizer.

    Returns:
        A dictionary of input IDs mappings.
    """
    logger.info(
        "Creating input ids mappings for interpolation of `embed_tokens` and `lm_head` layers."
    )
    input_ids_mappings = {}
    merge_tokenizer_vocab = merge_tokenizer.get_vocab()

    for model in merge_config.models + [merge_config.base_model]:
        vocab = all_tokenizers[model].get_vocab()
        vocab_size = get_vocab_size(
            model=model,
            trust_remote_code=merge_config.hf_hub_settings.trust_remote_code,
        ) or len(vocab)

        model_input_ids_mappings = {}
        for token, new_input_id in merge_tokenizer_vocab.items():
            old_input_id = vocab.get(token, -1)
            if old_input_id >= vocab_size:
                raise RuntimeError(
                    f"{model} token {token} has input id {old_input_id} > {vocab_size-1} due to trimming or modification."
                )
            model_input_ids_mappings[new_input_id] = old_input_id

        assert (
            len(merge_tokenizer_vocab) == len(model_input_ids_mappings)
        ), "Lengths of merge_tokenizer_vocab and model_input_ids_mappings must be equal."

        input_ids_mappings[model] = model_input_ids_mappings

    return input_ids_mappings
