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

    file_list: List[str]
    shard_filelist: List[str]

    # TODO: create compat mode
    hf_siblings: Optional[List[RepoSibling]]
    hf_safetensors: Optional[SafeTensorsInfo]

    # keep as is 
    hf_author: Optional[str]
    hf_created_at: Optional[datetime]
    hf_last_modified: Optional[datetime]
    hf_private: bool
    hf_gated: Optional[Literal["auto", "manual", False]]
    hf_disabled: Optional[bool]
    hf_library_name: Optional[str]
    hf_tags: List[str]
    hf_pipeline_tag: Optional[str]
    hf_mask_token: Optional[str]
    hf_card_data: Optional[ModelCardData]
    hf_widget_data: Optional[Any]
    hf_model_index: Optional[Dict]
    hf_transformers_info: Optional[TransformersInfo]
    hf_data: Optional[ModelInfo]

    # convenience checks
    has_config: bool = has_config_json(file_list)
    has_vocab: bool = has_tokenizer_file(file_list)
    has_tokenizer_config: bool = has_tokenizer_config(file_list)
    has_pytorch_bin_index: bool = has_pytorch_bin_index(file_list)
    has_safetensor_files: bool = has_safetensors_files(file_list)
    has_pytorch_bin_files: bool = has_pytorch_bin_files(file_list)
    has_adapter: bool = has_adapter_files(file_list)

    def __init__(self):
        # try to fetch the ModelInfo with id
        # if not id, sha, config from local
        # convert siblings and safetensors info
        self.siblings = 

        # add checks
        file_list = [f.rfilename for f in self.hf_siblings]
        self.file_list = file_list
        self.has_config = has_config_json(file_list)
        self.has_vocab = has_tokenizer_file(file_list)
        self.has_tokenizer_config = has_tokenizer_config(file_list)
        self.has_pytorch_bin_index = has_pytorch_bin_index(file_list)
        self.has_safetensor_files = has_safetensors_files(file_list)
        self.has_pytorch_bin_files = has_pytorch_bin_files(file_list)
        self.has_adapter = has_adapter_files(file_list)


def load_model_info(path_or_id):
    try:
        # get the model information from HF if available
        hf: ModelInfo = huggingface_hub.hf_api.repo_info(
            repo_id=path_or_id, repo_type="model", files_metadata=True
        )
        model_metadata = ModelMetadata(hf=hf, hf_exists=True)
    except huggingface_hub.hf_api.RepositoryNotFoundError:
        model_metadata = ModelMetadata(
            hf=None, custom=CustomModelInfo()
        )
    checks: Checks = Checks(model_metadata)
    model_metadata.checks = checks

    return model_metadata