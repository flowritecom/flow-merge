from typing import Dict, Any, Type
from flow_merge.lib import config
from flow_merge.lib.file_io import FileRepository
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from flow_merge.lib.merger.merger import Merger
from flow_merge.lib.model.architecture import ModelArchitectureProvider
from flow_merge.lib.model.metadata import ModelMetadataService
from flow_merge.lib.model.service import ModelService
from flow_merge.lib.tensor.index import TensorIndexService
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.tokenizer import MergeTokenizerService

services: Dict[Type, object] = {}


def get(name: Type[Any]) -> Any:
    if name not in services:
        services[name] = _create(name)
    return services[name]

def _create(name: Type[Any]) -> object:
    if name is Merger:
        return Merger(config.app_config.get(), get(MergeTokenizerService), get(ModelMetadataService), get(ModelService),
                      get(ModelArchitectureProvider),
                      get(TensorRepository))
    elif name is MergeTokenizerService:
        return MergeTokenizerService(app_config=config.app_config.get())
    elif name is ModelMetadataService:
        return ModelMetadataService(app_config=config.app_config.get(), )
    elif name is ModelArchitectureProvider:
        return ModelArchitectureProvider(app_config=config.app_config.get())
    elif name is NormalizationRunner:
        return NormalizationRunner(get(ModelArchitectureProvider))
    elif name is ModelService:
        return ModelService(config=config.app_config.get(), tensor_index_service=get(TensorIndexService),
                            file_repository=get(FileRepository), tensor_repository=get(TensorRepository),
                            metadata_service=get(ModelMetadataService))
    elif name is TensorRepository:
        return TensorRepository()
    elif name is FileRepository:
        return FileRepository(config.app_config.get())
    elif name is TensorIndexService:
        return TensorIndexService()
    else:
        raise Exception(f"Requested service ${name} is not available, check DI container configuration")
