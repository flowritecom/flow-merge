from pydantic import BaseModel, Field, field_validator, ValidationError

PathOrId = str

# top-level yaml configuration
class MergeConfiguration(BaseModel):
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching models and tokenizers with the `transformers library.",
    )
    local_dir: Path = Field(
        default=Path("./models").resolve(),
        description="Directory for loading models from local.",
    )
    output_dir: Path = Field(
        default=Path("./merged_model").resolve(),
        description="Directory for saving the merged model, tokenizer, and metadata.",
    )
    tokenizer_mode: str = Field(
        default="base",
        description="Method for obtaining the tokenizer for the merged model. 'base' uses the base model's tokenizer, 'merged' uses the merged model's tokenizer. If tokenizers use different tokenizer, linear interpolation of embedding and lm head layers will be performed.",
    )
    tokenizer_interpolation_method: str = Field(
        default="linear",
        description="Method for interpolating the token embeddings and language model head layers. 'linear' performs a linear interpolation between the two models.",
    )

    base_model: Optional[PathOrId]


    @field_validator("cache_dir", "local_dir", "output_dir")
    def validate_cache_dir(cls, v):
        if v:
            v = Path(v).resolve()
            v.mkdir(parents=True, exist_ok=True)
            return v

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
