from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import huggingface_hub
from pydantic import BaseModel
from huggingface_hub.hf_api import (
    ModelInfo,
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

from flow_merge.lib.tensor_index import FileToTensorIndex 

from transformers import AutoConfig

## TODO: Make the creation of this class testable from +10 models
## TODO: Potentially place the checks here


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

    hf_siblings: Optional[List[RepoSibling]] = None
    hf_safetensors: Optional[SafeTensorsInfo] = None

    hf_author: Optional[str] = None
    hf_created_at: Optional[datetime] = None
    hf_last_modified: Optional[datetime] = None
    hf_private: bool = False
    hf_gated: Optional[Literal["auto", "manual", False]] = None
    hf_disabled: Optional[bool] = None
    hf_library_name: Optional[str] = None
    hf_tags: List[str] = []
    hf_pipeline_tag: Optional[str] = None
    hf_mask_token: Optional[str] = None
    hf_card_data: Optional[ModelCardData] = None
    hf_widget_data: Optional[Any] = None
    hf_model_index: Optional[Dict] = None
    hf_transformers_info: Optional[TransformersInfo] = None
    hf_data: Optional[ModelInfo] = None

    # convenience checks
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
            siblings = [sibling for sibling in self.hf_siblings]

            # SCENARIO 1 
            # Sibling -> FileMetadata
            # fn: sibling -> common_file_metadata_object
            # map over the siblings

            # SCENARIO 2
            # LocalDir -> FileMetadata
            # create this using some other way from local data
            search_dir = model_dir / id

FileMetadata = str
def sibling_to_metadata(sibling: RepoSibling) -> FileMetadata:

    # RepoSibling
    # rfilename: str
    # size: Optional[int] = None
    # blob_id: Optional[str] = None
    # lfs: Optional[BlobLfsInfo] = None

    # BlobLfsInfo
    # size: int
    # sha256: str
    # pointer_size: int


def load_model_info(path_or_id):
    try:
        # get the model information from HF if available
        hf: ModelInfo = huggingface_hub.hf_api.repo_info(
            repo_id=path_or_id, repo_type="model", files_metadata=True
        )
        model_metadata = ModelMetadata(hf=hf, hf_exists=True)
    except huggingface_hub.hf_api.RepositoryNotFoundError:
        # NOT FOUND IN HF, INFERRING FROM LOCAL MODEL DIR
        # TODO: create id
        # TODO: create sha
        # TODO: create config dict  
            # check for config.json in the path_or_id  --> already checked in previous steps 
        
    
        config = AutoConfig.from_pretrained(path_or_id).to_dict()



        # TODO: create siblings equivalent
        # TODO: create hf siblings equivalent
        # TODO: create safetensors equivalent                                               