
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DirectorySettings(BaseModel):
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
