import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import Field, ValidationError, field_validator, BaseModel, ConfigDict, computed_field
from flow_merge.lib.validators.slice_validator import SliceValidator

logger = logging.getLogger(__name__)

class MergeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: str = Field()
    tokenizer_mode: str = Field(
        default="base",
        description="Method for obtaining the tokenizer for the merged model. 'base' uses the base model's tokenizer, 'merged' uses the merged model's tokenizer. If tokenizers use different tokenizer, linear interpolation of embedding and lm head layers will be performed.",
    )
    tokenizer_interpolation_method: str = Field(
        default="linear",
        description="Method for interpolating the token embeddings and language model head layers. 'linear' performs a linear interpolation between the two models.",
    )
    definition: List[Dict[str, Any]] = Field(
        description="Definition of the output model slices"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

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
            logger.warning(
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

    @field_validator("definition")
    def validate_definition(cls, v):
        slice_validator = SliceValidator()
        try:
            for s in v:
                slice_validator.validate(s)
        except ValueError as e:
            raise ValueError("definition", f"Slice validation error: {str(e)}")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "MergeConfig":
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
                return MergeConfig(**data)
        except Exception as e:
            raise Exception(f"Cannot open or decode provided yaml configuration: {str(e)}") from e

    @classmethod
    def from_json(cls, path: Path) -> "MergeConfig":
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                return MergeConfig(**data)
        except Exception as e:
            raise Exception(f"Cannot open or decode provided json configuration: {str(e)}") from e
