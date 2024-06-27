from enum import Enum
from typing import Dict

# from transformers library special tokens map
ADDITIONAL_SPECIAL_TOKENS_KEY = "additional_special_tokens"
CHUNK_SIZE = 64 * 1024  # 64KB


class DeviceIdentifier(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ConfigKeys(str, Enum):
    METHOD = "method"
    METHOD_GLOBAL_PARAMETERS = "method_global_parameters"
    BASE_MODEL = "base_model"
    MODELS = "models"
    MODEL = "model"
    WEIGHT = "weight"
    TOKENIZER = "tokenizer"
    DIRECTORY_SETTINGS = "directory_settings"
    HF_HUB_SETTINGS = "hf_hub_settings"
    DEVICE = "device"
    OUTPUT_DIR = "output_dir"
    SCALING_COEFFICIENT = "scaling_coefficient"
    NORMALIZE = "normalize"
    TOP_K = "top_k"
    P = "p"
    T = "t"
    MODE = "mode"
    INTERPOLATION_METHOD = "interpolation_method"
    CACHE_DIR = "cache_dir"
    LOCAL_DIR = "local_dir"
    TOKEN = "token"
    TRUST_REMOTE_CODE = "trust_remote_code"

# method_classes: Dict[MergeMethodIdentifier, TaskArithmetic | Linear | Slerp] = {
#     MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmetic,
#     MergeMethodIdentifier.MODEL_SOUP.value: Linear,
#     MergeMethodIdentifier.TIES_MERGING.value: TaskArithmetic,
#     MergeMethodIdentifier.DARE_TIES_MERGING.value: TaskArithmetic,
#     MergeMethodIdentifier.SLERP.value: Slerp,
# }

# method_configs: Dict[MergeMethodIdentifier, BaseMergeMethodSettings] = {
#     MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmeticSettings,
#     MergeMethodIdentifier.MODEL_SOUP.value: BaseMergeMethodSettings,
#     MergeMethodIdentifier.TIES_MERGING.value: TiesMergingSettings,
#     MergeMethodIdentifier.DARE_TIES_MERGING.value: DareTiesMergingSettings,
#     MergeMethodIdentifier.SLERP.value: SlerpSettings,
# }
