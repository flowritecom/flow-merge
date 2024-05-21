from typing import List, Optional, Any
from pydantic import BaseModel
from transformers import AutoConfig, PretrainedConfig

from flow_merge.lib.logger import get_logger
from flow_merge.lib.model_metadata import ModelMetadata, ModelMetadataService
from flow_merge.lib.shard import ShardFile
from flow_merge.lib.tensor_index import FileToTensorIndex
from flow_merge.lib.merge_settings import DirectorySettings

logger = get_logger(__name__)


class Model(BaseModel):
    path: str
    metadata: ModelMetadata
    file_to_tensor_index: FileToTensorIndex
    shards: List[ShardFile]
    config: Optional[PretrainedConfig]
    revision: Optional[str] = None
    trust_remote_code: bool = False

    @classmethod
    def from_path(cls, path: str, token: str, directory_settings: DirectorySettings):
        metadata_service = ModelMetadataService(
            token=token, directory_settings=directory_settings
        )
        metadata = metadata_service.load_model_info(path)
        return cls(
            path=path,
            metadata=metadata,
            file_to_tensor_index=FileToTensorIndex(metadata.file_metadata_list),
            shards=[ShardFile(file.filename) for file in metadata.file_metadata_list],
            config=AutoConfig.from_pretrained(path) if metadata.has_config else None,
            revision=metadata.sha,
            trust_remote_code=metadata.hf_transformers_info.trust_remote_code
            if metadata.hf_transformers_info
            else False,
        )

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
