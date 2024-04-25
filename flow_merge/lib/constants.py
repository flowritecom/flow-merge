from enum import Enum

# from transformers library special tokens map
ADDITIONAL_SPECIAL_TOKENS_KEY = "additional_special_tokens"


class DeviceIdentifier(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class MergeMethodIdentifier(str, Enum):
    ADDITION_TASK_ARITHMETIC = "addition-task-arithmetic"
    TIES_MERGING = "ties-merging"
    SLERP = "slerp"
    DARE_TIES_MERGING = "dare-ties-merging"
    MODEL_SOUP = "model-soup"
    PASSTHROUGH = "passthrough"


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
