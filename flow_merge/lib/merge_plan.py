import datetime
import hashlib
import json
from pathlib import Path
from typing import List, Any, Optional
from pydantic import BaseModel, computed_field, Field
from flow_merge.lib.loaders.normalizer import NormalizationRunner, NormalizedSlice
from flow_merge.lib.merge_config import MergeConfig


class MergePlan(BaseModel):
    created_at: datetime.datetime
    name: str
    base_model: str
    tokenizer_mode: str
    tokenizer_interpolation_method: str
    slices: Optional[List[NormalizedSlice]] = None
    raw_config: Optional[dict]
    lib_version: str = Field(
        default="0.0.1"  # fixme use actual version from appropriate source
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: MergeConfig, normalization_runner: NormalizationRunner) -> "MergePlan":
        slices = normalization_runner.normalize({
            "base_model": config.base_model,
            "definition": config.definition,
        })
        return cls(
            created_at=datetime.datetime.now(),
            name=config.name,
            base_model=config.base_model,
            tokenizer_mode=config.tokenizer_mode,
            tokenizer_interpolation_method=config.tokenizer_interpolation_method,
            slices=slices,
            raw_config=config.model_dump()
        )

    @classmethod
    def from_file(cls, file_path: Path) -> "MergePlan":
        with open(file_path, "rb") as f:
            parsed = json.load(f)
            return cls(**parsed)

    @computed_field
    @property
    def sha(self) -> str:
        return hashlib.md5(
            {
                "base_model": self.base_model,
                "tokenizer_mode": self.tokenizer_mode,
                "tokenizer_interpolation_method": self.tokenizer_interpolation_method,
                "slices": self.slices,
                "lib_version": self.lib_version,
            }.__str__().encode("utf-8")
        ).hexdigest()

    def to_json(self):
        return self.model_dump_json(indent=4, exclude_none=True)
