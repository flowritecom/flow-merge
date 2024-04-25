from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from flow_merge.lib.constants import ADDITIONAL_SPECIAL_TOKENS_KEY
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import MergeConfig, Model
from flow_merge.lib.model import Model

logger = get_logger(__name__)


@dataclass
class Tokenizer:
    tokenizer: PreTrainedTokenizerBase
    input_ids_mappings: Optional[Dict[Model, Dict[int, int]]]


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
    is_there_different_vocab = False

    for tokenizer_a, tokenizer_b in combinations(tokenizers.items(), 2):
        model_a, tokenizer_a = tokenizer_a
        model_b, tokenizer_b = tokenizer_b

        vocab_a = tokenizer_a.get_vocab()
        vocab_b = tokenizer_b.get_vocab()
        special_tokens_a = tokenizer_a.special_tokens_map
        special_tokens_b = tokenizer_b.special_tokens_map
        added_tokens_encoder_a = tokenizer_a.added_tokens_encoder
        added_tokens_encoder_b = tokenizer_b.added_tokens_encoder

        if vocab_a != vocab_b:
            is_there_different_vocab = True
            logger.info(
                f"Tokenizer for model {model_a} has different vocab compared to model {model_b}."
            )

        if special_tokens_a != special_tokens_b:
            is_there_different_vocab = True
            logger.info(
                f"Tokenizer for model {model_a} has different special tokens compared to model {model_b}."
            )

        if added_tokens_encoder_a != added_tokens_encoder_b:
            is_there_different_vocab = True
            logger.info(
                f"Tokenizer for model {model_a} has different added tokens encoder compared to model {model_b}."
            )

    return is_there_different_vocab


def construct_merged_tokenizer(
    base_model: Model, tokenizers: Dict[Model, PreTrainedTokenizerBase]
) -> PreTrainedTokenizerBase:
    """
    Constructs a tokenizer with a vocab that contains all tokens from all models involved in the merge.

    Args:
        base_model: The base model.
        tokenizers: A dictionary mapping models to their respective tokenizers.

    Returns:
        The merged tokenizer.

    Raises:
        None

    """
    # TODO - compare with m and recheck exceptions ???
    merged_vocab = {}
    merged_added_tokens = {}
    duplicate_added_tokens = set()
    merged_special_tokens = {}

    for model, tokenizer in tokenizers.items():
        if model == base_model:
            continue
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        added_tokens = tokenizer.added_tokens_decoder
        special_tokens = tokenizer.special_tokens_map

        for token, input_id in vocab.items():
            if input_id >= vocab_size:
                # Due to vocabulary size mismatch or vocabulary trimming
                raise RuntimeError(
                    f"Token {token} has input_id {input_id} which is greater than the vocab_size {vocab_size} of {model} tokenizer."
                )

            if token not in added_tokens and token not in merged_vocab:
                merged_vocab[token] = len(merged_vocab)

        for input_id, added_token in added_tokens.items():
            token = added_token.content
            if input_id >= vocab_size:
                raise RuntimeError(
                    f"Added token {token} has input_id {input_id} which is greater than the vocab_size {vocab_size} of {model} tokenizer."
                )

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

        # * SPECIAL TOKENS
        for special_token_type, special_token in special_tokens.items():
            if special_token_type == ADDITIONAL_SPECIAL_TOKENS_KEY and isinstance(
                special_token, list
            ):
                if special_token_type not in merged_special_tokens:
                    merged_special_tokens[special_token_type] = []
                merged_special_tokens[special_token_type].extend(special_token)
            else:
                if special_token_type not in merged_special_tokens:
                    merged_special_tokens[special_token_type] = special_token

    base_tokenizer = tokenizers[base_model]
    base_vocab = base_tokenizer.get_vocab()

    base_vocab_set = set(base_vocab)
    base_added_tokens_set = set(
        [token.content for token in base_tokenizer.added_tokens_decoder.values()]
    )
    merged_added_tokens_set = set(merged_added_tokens)

    merged_tokenizer = deepcopy(base_tokenizer)

    del base_tokenizer

    tokens_to_add = [
        token
        for token in merged_vocab
        if token not in merged_added_tokens_set and token not in base_vocab_set
    ]
    merged_tokenizer.add_tokens(tokens_to_add)

    tokens_to_add_with_settings = [
        merged_added_tokens[token]
        for token in merged_added_tokens
        if token not in base_added_tokens_set
    ]
    merged_tokenizer.add_tokens(tokens_to_add_with_settings)

    # SPECIAL TOKENS MERGE TOKENIZER
    # ! If conflict -> The last one takes priority
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

    Raises:
        NotImplemented: If the tokenizer method is not "linear".
        AttributeError: If the tokenizer source is invalid.

    """
    tokenizer_settings = merge_config.tokenizer_settings

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

    del tokenizer

    # Check if any tokenizer has different tokens or vocab
    logger.info(
        "Checking if there are differences in tokens or vocab among tokenizers..."
    )
    is_there_different_vocab = check_tokenizers_for_differences(
        tokenizers=all_tokenizers
    )

    if not is_there_different_vocab:
        logger.info(
            f"No differences in tokens or vocab among tokenizers. Using {merge_config.base_model.path} for the tokenizer."
        )
        return Tokenizer(
            tokenizer=all_tokenizers[merge_config.base_model], input_ids_mappings=None
        )

    logger.info(
        "Different tokens or vocab among tokenizers. Building the tokenizer for the merged model."
    )
    if tokenizer_settings.mode == "base":
        merge_tokenizer = all_tokenizers[merge_config.base_model]
    else:
        merge_tokenizer = construct_merged_tokenizer(
            merge_config.base_model, tokenizers=all_tokenizers
        )

    merge_tokenizer_vocab = (
        merge_tokenizer.get_vocab()
    )  # Tokenizer to be used by the merged model

    logger.info(
        "Creating input ids mappings for interpolation of `embed_tokens` and `lm_head` layers."
    )
    input_ids_mappings = {}
    for model in merge_config.models + [merge_config.base_model]:
        vocab = all_tokenizers[model].get_vocab()
        vocab_size = get_vocab_size(
            model=model,
            trust_remote_code=merge_config.hf_hub_settings.trust_remote_code,
        ) or len(vocab)

        model_input_ids_mappings = {}
        for token in merge_tokenizer_vocab:
            new_input_id = merge_tokenizer_vocab[token]
            if token not in vocab:
                model_input_ids_mappings[new_input_id] = -1  # Token does not exist
                continue
            old_input_id = vocab[token]
            if old_input_id >= vocab_size:
                raise RuntimeError(
                    f"{model} token {token} has input id {old_input_id} > {vocab_size-1} due to trimming or modification."
                )
            model_input_ids_mappings[new_input_id] = old_input_id  # * {3845: 4759, ...}

        assert (
            len(merge_tokenizer_vocab) == len(model_input_ids_mappings)
        ), "Lengths of merge_tokenizer_vocab and model_input_ids_mappings must be equal."

        input_ids_mappings[model] = model_input_ids_mappings

    return Tokenizer(tokenizer=merge_tokenizer, input_ids_mappings=input_ids_mappings)
