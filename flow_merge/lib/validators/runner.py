from typing import Optional, Dict, Any

from flow_merge.lib.validators._method_settings import MethodSettings
from flow_merge.lib.validators._directory_settings import DirectorySettings
from flow_merge.lib.validators._tokenizer_settings import TokenizerSettings
from flow_merge.lib.validators._model_settings import ModelSettings
from flow_merge.lib.validators._hf_hub_settings import HfHubSettings
from flow_merge.lib.validators._hardware_settings import HardwareSettings

# FIXME: run Normalizer before this
# FIXME: have the ability to choose NormalizationRunner, ValidationRunner, LegalityCheckRunner
class ValidationRunner:
    def __init__(self, raw_data: Dict[str, Any], env, logger):
        self.env = env
        self.logger = logger
        self.raw_data = raw_data
        self.merge_method, self.method_global_parameters = self.validate(
            MethodSettings, ["method", "method_global_parameters"]
        )._unpack()
        self.directory_settings = self.validate(DirectorySettings, ['directory_settings'])
        self.tokenizer_settings = self.validate(TokenizerSettings, ['tokenizer'])
        self.base_model, self.models = self.validate(ModelSettings, ['base_model', 'models'])._unpack()
        self.trust_remote_code = self.validate(HfHubSettings, ['trust_remote_code'])
        self.device = self.validate(HardwareSettings, ['device'])._unpack()


    def validate(self, settings_class, keys):
        return settings_class(
            **{k: self.raw_data[k] for k in keys if k in self.raw_data}
        )
