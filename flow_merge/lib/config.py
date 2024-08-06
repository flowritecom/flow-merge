import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ValidationError
import logging
from huggingface_hub import login, logout


class DeviceIdentifier(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


logger = logging.getLogger(__name__)


class ApplicationConfig(BaseModel):
    device: DeviceIdentifier = Field(default=DeviceIdentifier.CPU)
    hf_token: str = Field(..., default_factory=lambda: os.getenv("HF_TOKEN"))
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching models and tokenizers with the transformers library.",
    )
    local_dir: Path = Field(
        default=Path("./models").resolve(),
        description="Directory for loading models from local.",
    )
    output_dir: Path = Field(
        default=Path("./merged_model").resolve(),
        description="Directory for saving the merged model, tokenizer, and metadata.",
    )

    def __post_init__(self):
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        logger.info(
            "HF_HUB_DISABLE_IMPLICIT_TOKEN set to 1 to disable implicit token authentication."
        )

    def set_hf_token(self, token: str):
        self.hf_token = token

    def set_device(self, device: str):
        self.device = device

    @field_validator("hf_token")
    def validate_hf_token(cls, v):
        if v:
            token_pattern = r"^hf_[a-zA-Z0-9]+$"
            if not re.match(token_pattern, v):
                logger.warning(
                    f"Invalid Hugging Face Hub token format. HF token should be of the form '{token_pattern}'."
                )
                raise ValueError("Invalid token format")

            try:
                login(token=v)
                logout(token=v)
            except Exception as e:
                logger.warning(
                    f"Failed to login to the Hugging Face Hub with the provided token: {e}"
                )
                raise ValueError("Failed to authenticate with the provided token")
        return v

    @field_validator("cache_dir")
    def validate_cache_dir(cls, v):
        if v:
            v = Path(v).resolve()
            v.mkdir(parents=True, exist_ok=True)
            return v

    @field_validator("local_dir")
    def validate_local_dir(cls, v):
        v = Path(v).resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("output_dir")
    def validate_output_dir(cls, v):
        v = Path(v).resolve()
        v.mkdir(parents=True, exist_ok=True)
