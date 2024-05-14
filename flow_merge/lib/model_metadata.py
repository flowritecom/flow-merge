import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import huggingface_hub
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoConfig
from huggingface_hub.hf_api import (
    ModelInfo, BlobLfsInfo
)
from flow_merge.lib.io import (
    has_adapter_files,
    has_config_json,
    has_pytorch_bin_files,
    has_pytorch_bin_index,
    has_safetensors_files,
    has_tokenizer_config,
    has_tokenizer_file,
)

from huggingface_hub.hf_api import (
    ModelInfo,
    RepoSibling,
    SafeTensorsInfo,
    ModelCardData,
    TransformersInfo,
)
from flow_merge.lib.logger import get_logger

# from flow_merge.lib.tensor_index import FileToTensorIndex

## TODO: Make the creation of this class testable from +10 models
## TODO: Potentially place the checks here

logger = get_logger(__name__)


class LocalRepoSibling(BaseModel):
    rfilename: str
    size: Optional[int] = None
    oid: Optional[str] = None


class FileMetadata(BaseModel):
    # Defer to when we have locally
    filename: str
    oid: Optional[str] = None
    blob_id: Optional[str] = None
    size: Optional[int] = None
    lfs: Optional[BlobLfsInfo] = None


class ModelMetadata(BaseModel):
    """
    Contains information about a model.

    Attributes:
        id (`str`):
            ID of model.
        author (`str`, *optional*):
            Author of the model.
        sha (`str`, *optional*):
            Repo SHA at this particular revision.
        created_at (`datetime`, *optional*):
            Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`,
            corresponding to the date when we began to store creation dates.
        last_modified (`datetime`, *optional*):
            Date of last commit to the repo.
        private (`bool`):
            Is the repo private.
        disabled (`bool`, *optional*):
            Is the repo disabled.
        gated (`Literal["auto", "manual", False]`, *optional*):
            Is the repo gated.
            If so, whether there is manual or automatic approval.
        library_name (`str`, *optional*):
            Library associated with the model.
        tags (`List[str]`):
            List of tags of the model. Compared to `card_data.tags`, contains extra tags computed by the Hub
            (e.g. supported libraries, model's arXiv).
        pipeline_tag (`str`, *optional*):
            Pipeline tag associated with the model.
        mask_token (`str`, *optional*):
            Mask token used by the model.
        widget_data (`Any`, *optional*):
            Widget data associated with the model.
        model_index (`Dict`, *optional*):
            Model index for evaluation.
        config (`Dict`, *optional*):
            Model configuration.
        transformers_info (`TransformersInfo`, *optional*):
            Transformers-specific info (auto class, processor, etc.) associated with the model.
        card_data (`ModelCardData`, *optional*):
            Model Card Metadata  as a [`huggingface_hub.repocard_data.ModelCardData`] object.
        siblings (`List[RepoSibling]`):
            List of [`huggingface_hub.hf_api.RepoSibling`] objects that constitute the model.
        safetensors (`SafeTensorsInfo`, *optional*):
            Model's safetensors information.
    """
    id: str
    sha: Optional[str]
    config: Optional[Dict]

    files_metadata: Optional[List[FileMetadata]] = []
    file_list: Optional[List[str]] = None

    hf_siblings: Optional[List[Union[RepoSibling, LocalRepoSibling]]] = Field(alias="siblings", default=None)
    hf_safetensors: Optional[SafeTensorsInfo] = Field(alias="safetensors", default=None)

    hf_author: Optional[str] = Field(alias="author", default=None)
    hf_created_at: Optional[datetime] = Field(alias="created_at", default=None)
    hf_last_modified: Optional[datetime] = Field(alias="last_modified", default=None)
    hf_private: bool = Field(alias="private", default=None)
    hf_gated: Optional[Literal["auto", "manual", False]] = Field(alias="gated", default=None)
    hf_disabled: Optional[bool] = Field(alias="disabled", default=None)
    hf_library_name: Optional[str] = Field(alias="library_name", default=None)
    hf_tags: List[str] = Field(alias="tags", default=[])
    hf_pipeline_tag: Optional[str] = Field(alias="pipeline_tag", default=None)
    hf_mask_token: Optional[str] = Field(alias="mask_token", default=None)
    hf_card_data: Optional[ModelCardData] = Field(alias="card_data", default=None)
    hf_widget_data: Optional[Any] = Field(alias="widget_data", default=None)
    hf_model_index: Optional[Dict] = Field(alias="model_index", default=None)
    hf_transformers_info: Optional[TransformersInfo] = Field(alias="transformers_info", default=None)
    hf_data: Optional[ModelInfo] = Field(alias="data", default=None)

    # convenience checks
    hf_exists: bool = True
    has_config: bool = False
    has_vocab: bool = False
    has_tokenizer_config: bool = False
    has_pytorch_bin_index: bool = False
    has_safetensor_files: bool = False
    has_pytorch_bin_files: bool = False
    has_adapter: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_checks()
        self.create_file_metadata()


    def update_checks(self):
        if self.hf_siblings:
            self.file_list = [sibling.rfilename for sibling in self.hf_siblings]
            self.has_config = has_config_json(self.file_list)
            self.has_vocab = has_tokenizer_file(self.file_list)
            self.has_tokenizer_config = has_tokenizer_config(self.file_list)
            self.has_pytorch_bin_index = has_pytorch_bin_index(self.file_list)
            self.has_safetensor_files = has_safetensors_files(self.file_list)
            self.has_pytorch_bin_files = has_pytorch_bin_files(self.file_list)
            self.has_adapter = has_adapter_files(self.file_list)
    
    def create_file_metadata(self):
        if self.hf_siblings:
            for sibling in self.hf_siblings:

                # TODO Figure out a way to generate a content hash (oid) for HF - files aren't locally loaded yet

                if self.hf_exists:
                    self.files_metadata.append(
                        FileMetadata(
                            blob_id=sibling.blob_id,
                            filename=sibling.rfilename,
                            size=sibling.size,
                            lfs=sibling.lfs   
                        )
                    )
                
                else:
                    self.files_metadata.append(
                        FileMetadata(
                            oid=sibling.oid,
                            filename=sibling.rfilename,
                            size=sibling.size,
                        )
                    )


def generate_oid(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        # TODO: confirm git lfs default chunk size (right now ~65KB)
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_model_info(path_or_id):
    try:
        # get the model information from HF if available
        hf_model_info: ModelInfo = huggingface_hub.hf_api.repo_info(
            repo_id=path_or_id,
            repo_type="model",
            files_metadata=True,
            token="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"
        )

        return ModelMetadata(**hf_model_info.__dict__)
    
    except huggingface_hub.hf_api.RepositoryNotFoundError:
        # NOT FOUND IN HF, INFERRING FROM LOCAL MODEL DIR 
        logger.info(f"Model not found in Hugging face. Inferring from local model directory.")
        siblings = []

        # TODO: get the base path from env
        base_path = "../models/"
        path_to_model = Path(base_path + path_or_id).resolve()

        if path_to_model.exists():
            for file_path in path_to_model.glob("*"):
                oid = generate_oid(file_path)
                filename = file_path.name
                size = file_path.stat().st_size

                siblings.append(
                    LocalRepoSibling(
                        oid=oid, 
                        size=size,
                        rfilename=filename,
                    )
                )
            
            try:
                config = AutoConfig.from_pretrained(path_to_model).to_dict()
            except EnvironmentError as e:
                # TODO update logging; fallback?
                logger.warn(f"Error while fetching config for local model: {e}")

            return ModelMetadata(
                id=path_or_id,
                sha=None,
                siblings=siblings,
                config=config,
                hf_exists=False
            )
        
        else:
            logger.warn("Model not found locally, cannot create model metadata.")

