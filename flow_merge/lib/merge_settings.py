import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError, field_validator, BaseModel


class MergeSettings(BaseModel):
    tokenizer_mode: str = Field(
        default="base",
        description="Method for obtaining the tokenizer for the merged model. 'base' uses the base model's tokenizer, 'merged' uses the merged model's tokenizer. If tokenizers use different tokenizer, linear interpolation of embedding and lm head layers will be performed.",
    )
    tokenizer_interpolation_method: str = Field(
        default="linear",
        description="Method for interpolating the token embeddings and language model head layers. 'linear' performs a linear interpolation between the two models.",
    )
    output_dir: Path = Field(
        default=Path("./merged_model").resolve(),
        description="Directory for saving the merged model, tokenizer, and metadata.",
    )

    def _unpack(self):
        return self.tokenizer_mode, self.tokenizer_interpolation_method, self.output_dir

    @field_validator("tokenizer_mode")
    def validate_mode(cls, v):
        if v is not None and v not in ["base", "merged"]:
            raise ValidationError(
                "tokenizer_mode",
                f"Invalid tokenizer mode: {v}. Allowed modes are 'base' and 'merged'.",
            )
        return v

    @field_validator("tokenizer_interpolation_method")
    def validate_interpolation_method(cls, v):
        if not v:
            logging.info(
                "No interpolation method provided for tokenizer of the merged model. Defaulting to 'linear' in case interpolation of token embed and lm head layers is needed due to different vocabularies of tokenizers."
            )
            return v
        if v not in ["linear"]:
            raise ValidationError(
                "tokenizer_interpolation_method",
                f"Invalid interpolation method: '{v}'. Allowed methods are 'linear' only.",
            )
        else:
            return v

    @field_validator("output_dir")
    def validate_output_dir(cls, v):
        v = Path(v).resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v
