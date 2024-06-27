from pathlib import Path
from typing import List, NewType, Optional, Dict

from pydantic import BaseModel

from flow_merge.lib.logger import Logger
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.validators import DirectorySettings
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.model.metadata import ModelMetadataService, ModelMetadata
from flow_merge.lib.tensor.index import TensorIndexService
from flow_merge.lib.tensor.loader import ShardFile

ModelId = NewType("ModelId", str)


class ModelBase(BaseModel, arbitrary_types_allowed=True):

    @classmethod
    def from_path(cls):
        pass

    @classmethod
    def from_layers(cls):
        pass


class Model(ModelBase, arbitrary_types_allowed=True):
    id: ModelId
    path: Path
    metadata: ModelMetadata
    file_to_tensor_index: Optional[Dict]
    shards: List[ShardFile]

    is_partial: bool = False
    
    @staticmethod
    def _create_metadata(
        path: Path, 
        directory_settings: DirectorySettings,
        env: ApplicationConfig,
        logger: Logger
        ):
        metadata_service = ModelMetadataService(
            directory_settings=directory_settings,
            env=env,
            logger=logger
        )

        metadata = metadata_service.load_model_info(str(path))
        
        return metadata
    
    @classmethod
    def from_path(
        cls, 
        path: Path, 
        directory_settings: DirectorySettings, 
        env: ApplicationConfig, 
        logger: Logger
    ):
        metadata = cls._create_metadata(path, directory_settings, env, logger)

        model_id = ModelId(str(path))
        file_to_tensor_index = TensorIndexService.create_file_to_tensor_index(metadata)

        shards = ModelService.create_shard_files(
            model_metadata=metadata,
            device=env.device,
            layers_to_download=None
        )

        return cls(
            id=model_id,
            path=path.resolve(),
            metadata=metadata,
            file_to_tensor_index=file_to_tensor_index,
            shards=shards,
        )
    
    @classmethod
    def from_layers(
        cls, 
        layers_to_download, 
        path, 
        directory_settings,
        env: ApplicationConfig,
        logger: Logger
    ):
        metadata = cls._create_metadata(path, directory_settings, env, logger)

        model_id = ModelId(str(path))
        file_to_tensor_index = TensorIndexService.create_file_to_tensor_index(metadata)

        shards = ModelService.create_shard_files(
            model_metadata=metadata,
            device=env.device,
            layers_to_download=layers_to_download
        )

        return cls(
            id=model_id,
            path=path.resolve(),
            metadata=metadata,
            file_to_tensor_index=file_to_tensor_index,
            shards=shards,
            is_partial=False if metadata.has_adapter and file_to_tensor_index is None else True
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