from pathlib import Path
from typing import List, NewType, Optional, Dict

from pydantic import BaseModel
from transformers import PretrainedConfig

from flow_merge.lib.logger import Logger
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.validators import DirectorySettings
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.model.architecture import ModelArchitecture
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
    architecture: ModelArchitecture

    is_partial: bool = False
    
    @staticmethod
    def _create_metadata(
        path: Path, 
        directory_settings: DirectorySettings,
        ):
        print("Creating metadata: _create_metadata")
        metadata_service = ModelMetadataService(
            directory_settings=directory_settings,
        )

        metadata = metadata_service.load_model_info(str(path))
        
        return metadata
    
    @staticmethod
    def _create_architecture(
        metadata: ModelMetadata,
    ):
        print("Creating architecture: _create_architecture")
        try:
            config = PretrainedConfig.from_dict(metadata.config)
            return ModelArchitecture.from_config(config)
        except EnvironmentError as e:
            print(f"Error while fetching config for local model: {e}")
            
    
    @classmethod
    def from_path(
        cls, 
        path: Path, 
        directory_settings: DirectorySettings = DirectorySettings()
    ):
        print("Loading Model from path: from_path")
        metadata = cls._create_metadata(path, directory_settings)

        model_id = ModelId(str(path))
        file_to_tensor_index = TensorIndexService.create_file_to_tensor_index(metadata)

        shards = ModelService.create_shard_files(
            model_metadata=metadata,
            layers_to_download=None
        )

        architecture = cls._create_architecture(metadata)

        print("Creating the Model class: from_path")
        return cls(
            id=model_id,
            path=Path(directory_settings.local_dir, path).resolve(),
            metadata=metadata,
            file_to_tensor_index=file_to_tensor_index,
            shards=shards,
            architecture=architecture
        )
    
    @classmethod
    def from_layers(
        cls, 
        layers_to_download, 
        path, 
        directory_settings,
    ):
        print("Loading Model from layers: from_layers")
        metadata = cls._create_metadata(path, directory_settings)

        model_id = ModelId(str(path))
        file_to_tensor_index = TensorIndexService.create_file_to_tensor_index(metadata)

        shards = ModelService.create_shard_files(
            model_metadata=metadata,
            layers_to_download=layers_to_download
        )

        architecture = cls._create_architecture(metadata)

        print("Creating the Model class: from_layers")
        return cls(
            id=model_id,
            path=Path(directory_settings.local_dir, path).resolve(),
            metadata=metadata,
            file_to_tensor_index=file_to_tensor_index,
            shards=shards,
            is_partial=False if metadata.has_adapter and file_to_tensor_index is None else True,
            architecture=architecture
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
