from typing import Optional, Dict, Any
from pydantic import BaseModel, field_validator
from ...validators._model_settings import ModelSettings
from ...validators._method_settings import MethodSettings
from ...validators._tokenizer_settings import TokenizerSettings
from ..hash import create_content_hash

class MergeSettings(BaseModel):
    models: ModelSettings
    method: MethodSettings
    tokenizer: TokenizerSettings
    sha: Optional[str]

    @field_validator('sha', mode='after')
    def compute_sha(cls, values: Dict[str, Any]):
        # Convert all fields except 'sha' to a dictionary
        data_dict = {k: v for k, v in values.items() if k != 'sha'}
        content_hash = create_content_hash(data_dict)
        values['sha'] = content_hash
        return values

