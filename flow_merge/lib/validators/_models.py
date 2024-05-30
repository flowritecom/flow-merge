
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


PathOrId = str
class RawModelDict(BaseModel):
    path_or_id: PathOrId = Field(alias="model")
    weight: Optional[float] = None

class Models(BaseModel):
    base_model: Optional[PathOrId]
    models: List[RawModelDict]

    @field_validator("models")
    @classmethod
    def validate_models(cls, v) -> None:
        if len(v) < 2:
            raise ValidationError(
                "At least two models should be provided for merging. Either a base model "
                + "and a model or two models if the method provided does not use a base model."
            )
        return v
