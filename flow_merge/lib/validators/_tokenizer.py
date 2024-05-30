
from pydantic import BaseModel, Field, ValidationError, field_validator


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
