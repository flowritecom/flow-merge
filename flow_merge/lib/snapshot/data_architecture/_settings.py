from typing import Optional, Dict, Any
from pydantic import BaseModel, model_validator
from ...validators._model_settings import ModelSettings
from ...validators._method_settings import MethodSettings
from ...validators._tokenizer_settings import TokenizerSettings
from flow_merge.lib.validators._directory_settings import DirectorySettings
from flow_merge.lib.validators._hf_hub_settings import HfHubSettings
from ..hash import create_content_hash

class MergeSettings(BaseModel):
    # FIXME We might not have model settings here, with the normalized slices
    models: ModelSettings
    method: MethodSettings
    tokenizer_settings: TokenizerSettings
    directory_settings: DirectorySettings
    hf_hub_settings: HfHubSettings = HfHubSettings()
    sha: Optional[str]

    @model_validator(mode="after")
    def compute_sha(self):
        # Convert all fields except 'sha' to a dictionary
        data_json = self.model_dump_json()
        content_hash = create_content_hash(
            data=data_json, 
            is_json=True
        )
        self.sha = content_hash

        return self

