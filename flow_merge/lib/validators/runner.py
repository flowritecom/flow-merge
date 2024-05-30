from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from flow_merge.lib.validators._directory import DirectorySettings
from flow_merge.lib.validators._tokenizer import TokenizerSettings
from flow_merge.lib.validators._method import MethodSettings
from flow_merge.lib.validators._models import ModelSettings

directory_settings: Optional[DirectorySettings] = DirectorySettings()
tokenizer_settings: Optional[TokenizerSettings] = TokenizerSettings()


class ValidationRunner:
    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.merge_method, self.method_global_parameters = self.validate(
            MethodSettings, ["method", "method_global_parameters"]
        )._unpack()
        self.directory_settings = self.validate(DirectorySettings, ['directory_settings'])
        self.tokenizer_settings = self.validate(TokenizerSettings, ['tokenizer'])
        self.base_model, self.models = self.validate(ModelSettings, ['base_model', 'models'])._unpack()

    def validate(self, settings_class, keys):
        return settings_class(
            **{k: self.raw_data[k] for k in keys if k in self.raw_data}
        )
