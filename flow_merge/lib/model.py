from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError, field_validator
from transformers import AutoConfig, PretrainedConfig

from flow_merge.lib.constants import ConfigKeys, MergeMethodIdentifier
from flow_merge.lib.logger import get_logger
from flow_merge.lib.model_metadata import ModelMetadata
from flow_merge.lib.shard import ShardFile
from flow_merge.lib.tensor_index import FileToTensorIndex

logger = get_logger(__name__)


class Model(BaseModel, arbitrary_types_allowed=True):

    # check the other places
    path: str

    metadata: ModelMetadata

    # concrete
    file_to_tensor_index: FileToTensorIndex
    shards: List[ShardFile]

    # transfer to ModelMetadata
    config: Optional[PretrainedConfig]
    revision: Optional[str] = None
    trust_remote_code: bool = False

    # check if needed
    @classmethod
    def from_path(cls, path: str, trust_remote_code: bool = False):
        path_and_revision = cls.validate_path(path)
        if isinstance(path_and_revision, tuple):
            path, revision = path_and_revision
        else:
            revision = None
        config = cls._load_config(path, revision, trust_remote_code)
        return cls(
            path=path,
            config=config,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def _load_config(
        cls, path: str, revision: Optional[str], trust_remote_code: bool
    ) -> PretrainedConfig:
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config for model {path}") from e

    def __hash__(self):
        return hash((self.path, self.revision))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.path == other.path
        return False

    def __str__(self):
        if self.revision:
            return f"{self.path}@{self.revision}"
        return self.path

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        if isinstance(v, str):
            ats = v.count("@")
            if ats > 1:
                raise ValidationError("path", f"Invalid model path - multiple @: {v}")
            elif ats == 1:
                v, revision = v.split("@")
                return v, revision
            else:
                return v
        else:
            raise ValidationError(
                "path", f"Expected path to be a string, got {type(v)}"
            )
