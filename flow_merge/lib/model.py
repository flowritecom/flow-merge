from pathlib import Path
from typing import List, Optional, NewType
from pydantic import BaseModel

from flow_merge.lib.logger import get_logger
from flow_merge.lib.model_metadata import ModelMetadata, ModelMetadataService
from flow_merge.lib.tensor_loader import (
    ShardFile,
    ModelService,
    FileToTensorIndex,
    TensorIndexService,
)
from flow_merge.lib.merge_settings import DirectorySettings

logger = get_logger(__name__)

ModelId = NewType("ModelId", str)


class Model(BaseModel, arbitrary_types_allowed=True):
    id: ModelId
    path: Path
    metadata: ModelMetadata
    file_to_tensor_index: Optional[FileToTensorIndex]
    shards: List[ShardFile]

    @classmethod
    def from_path(
        cls, path: Path, directory_settings: DirectorySettings
    ):
        metadata_service = ModelMetadataService(directory_settings=directory_settings)
        metadata = metadata_service.load_model_info(str(path))

        model_id = ModelId(str(path))
        file_to_tensor_index = TensorIndexService.create_file_to_tensor_index(metadata)

        if file_to_tensor_index is None and metadata.file_metadata_list:
            relevant_files = [
                file
                for file in metadata.file_metadata_list
                if file.filename.endswith((".safetensors", ".bin"))
            ]
            single_file = (
                relevant_files[0].filename
                if relevant_files
                else metadata.file_metadata_list[0].filename
            )
            shard = ModelService.create_shard_file(
                path, metadata.id, "cpu", single_file
            )
            shards = [shard]
        else:
            shards = ModelService.gather_shard_files(
                file_to_tensor_index, path, metadata.id, "cpu"
            )

        return cls(
            id=model_id,
            path=path.resolve(),
            metadata=metadata,
            file_to_tensor_index=file_to_tensor_index,
            shards=shards,
        )

    def __hash__(self):
        return hash((self.id, self.metadata.sha))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.id == other.id and self.metadata.sha == other.metadata.sha
        return False

    def __str__(self):
        if self.metadata.sha:
            return f"{self.id}@{self.metadata.sha}"
        return str(self.id)
