from typing import Optional
import re
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path
import os

from huggingface_hub import login

from flow_merge.lib.logger import get_logger
from flow_merge.lib.config import config

logger = get_logger(__name__)

PathOrId = str


class RawModelDict(BaseModel):
    path_or_id: PathOrId = Field(alias="model")
    weight: Optional[float] = None


# TODO: we can pull validation here
class MethodGlobalParameters(BaseModel):
    scaling_coefficient: Optional[float] = None
    normalize: Optional[bool] = None
    p: Optional[float] = None
    t: Optional[float] = None
    top_k: Optional[float] = None


class TokenizerSettings(BaseModel):
    mode: str = Field(
        default="base",
        description="Method for obtaining the tokenizer for the merged model. 'base' uses the base model's tokenizer, 'merged' uses the merged model's tokenizer. If tokenizers use different tokenizer, linear interpolation of embedding and lm head layers will be performed.",
    )
    interpolation_method: str = Field(
        default="linear",
        description="Method for interpolating the token embeddings and language model head layers. 'linear' performs a linear interpolation between the two models.",
    )

    @field_validator("mode")
    def validate_mode(cls, v):
        if v is not None and v not in ["base", "merged"]:
            raise ValidationError(
                "mode",
                f"Invalid tokenizer mode: {v}. Allowed modes are 'base' and 'merged'.",
            )
        return v

    @field_validator("interpolation_method")
    def validate_interpolation_method(cls, v):
        if not v:
            logger.info(
                "No interpolation method provided for tokenizer of the merged model. Defaulting to 'linear' in case interpolation of token embed and lm head layers is needed due to different vocabularies of tokenizers."
            )
            return v
        if v not in ["linear"]:
            raise ValidationError(
                "interpolation_method",
                f"Invalid interpolation method: '{v}'. Allowed methods are 'linear' only.",
            )
        else:
            return v


class HfHubSettings(BaseModel):
    token: Optional[str] = Field(
        default=config.hf_token,
        description="Hugging Face API token for downloading and pushing models from the Hugging Face Hub.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading models from the Hugging Face Hub or not.",
    )

    @field_validator("token")
    def validate_token(cls, v):
        if v:
            # Check the token format using a regular expression
            token_pattern = r"^hf_[a-zA-Z0-9]+$"
            if not re.match(token_pattern, v):
                logger.warning(
                    f"Invalid Hugging Face Hub token format. HF token should be of the form '{token_pattern}'."
                )
                return v

            # Attempt to login with the provided token
            try:
                login(token=v)
            except Exception as e:
                logger.warning(
                    f"Failed to login to the Hugging Face Hub with the provided token: {e}"
                )
        return v


class DirectorySettings(BaseModel):
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching models and tokenizers with the `transformers library.",
    )
    local_dir: Path = Field(
        default=Path("./models"), description="Directory for loading models from local."
    )
    output_dir: Path = Field(
        default=Path("./merged_model"),
        description="Directory for saving the merged model, tokenizer, and metadata.",
    )

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
        return v
