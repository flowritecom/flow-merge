from datetime import datetime
from pathlib import Path
from huggingface_hub.hf_api import (
    BlobLfsInfo,
    ModelCardData,
    ModelInfo,
    RepoSibling,
    SafeTensorsInfo,
    TransformersInfo,
)
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from flow_merge.lib.validators import DirectorySettings
from flow_merge.lib.model.metadata.file_metadata import FileMetadata

class ModelMetadata(BaseModel):
    id: str
    sha: Optional[str]
    config: Optional[Dict]
    file_metadata_list: List[FileMetadata] = Field(default_factory=list)
    file_list: List[str] = Field(default_factory=list)
    safetensors_info: Optional[SafeTensorsInfo] = Field(
        alias="safetensors", default=None
    ) 

    relative_path: Optional[Path] = None
    absolute_path: Optional[Path] = None

    # TODO: How are we passing directory settings via the snapshot?
    directory_settings: Optional[DirectorySettings] = None

    hf_author: Optional[str] = Field(alias="author", default=None)
    hf_created_at: Optional[datetime] = Field(alias="created_at", default=None)
    hf_last_modified: Optional[datetime] = Field(alias="last_modified", default=None)
    hf_private: bool = Field(alias="private", default=None)
    hf_gated: Optional[Literal["auto", "manual", False]] = Field(
        alias="gated", default=None
    )
    hf_disabled: Optional[bool] = Field(alias="disabled", default=None)
    hf_library_name: Optional[str] = Field(alias="library_name", default=None)
    hf_tags: List[str] = Field(alias="tags", default_factory=list)
    hf_pipeline_tag: Optional[str] = Field(alias="pipeline_tag", default=None)
    hf_mask_token: Optional[str] = Field(alias="mask_token", default=None)
    hf_card_data: Optional[ModelCardData] = Field(alias="card_data", default=None)
    hf_widget_data: Optional[Any] = Field(alias="widget_data", default=None)
    hf_model_index: Optional[Dict] = Field(alias="model_index", default=None)
    hf_transformers_info: Optional[TransformersInfo] = Field(
        alias="transformers_info", default=None
    )
    hf_data: Optional[ModelInfo] = Field(alias="data", default=None)

    hf_exists: bool = True
    has_config: bool = False
    has_vocab: bool = False
    has_tokenizer_config: bool = False
    has_pytorch_bin_index: bool = False
    has_safetensors_index: bool = False
    has_safetensor_files: bool = False
    has_pytorch_bin_files: bool = False
    has_adapter: bool = False