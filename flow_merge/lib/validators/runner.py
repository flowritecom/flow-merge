from typing import Optional, Dict, Any

from flow_merge.lib.validators._method_settings import MethodSettings
from flow_merge.lib.validators._directory_settings import DirectorySettings
from flow_merge.lib.validators._tokenizer_settings import TokenizerSettings
from flow_merge.lib.validators._model_settings import ModelSettings
from flow_merge.lib.validators._hf_hub_settings import HfHubSettings
from flow_merge.lib.validators._hardware_settings import HardwareSettings


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

# FIXME: can base_level key be at the top level -> fix this in Normalization also if yes

# FIXME: Create
# PW Config Loader -> FirstValidation -> Normalization ->
#
#                Snapshot <-> Legality Check
#
#          Snapshot:
#               DOES
#               - creating fingerprint (identity) data of everything
#               - also stores that data
#               - set that data online (?) into a db at some point,
#                 or be fed to a process that sets it
#               - legal checking
#                   - M: snapshot should be truthful and legal
#                   - Plan is locally instantiated snapshot with memory addresses
#                     that should not be seeked to reproduce
#               Creator:
#                  NormalizedForm ->
#                       LegalityCheck ->
#                           ShaCreation -> Snapshot
#
#               DOES NOT DO
#               - loading (is part of a loader)
#               - does not fetch more than addresses and metadata
#                   - does not store file content
#
#
#           Loader takes as argument
#               validation_runner
#               normalization_runner
#               snapshot_runner
#
#
#
#
#
#          WE WANT TO ARRIVE IN CREATION OF A PLAN AND EXECUTABLE SET
#
#          complete in terms addressing (or at least able to be completed/fetched)
#          legal in terms of operations and what is used
#          reproducible (SHAs created -> if we did it again, are we able to tell?)
#    -->   Plan object exists -> needs to be observable and mutable
#
#               Executable set ( ?? + ?device|other? )
#               -> Plan materialization
#                   -> create tokenizer
#                   -> create models
#                   -> create necessary engines (like RunPod)
#
#
#
#
#
#
#
#
# TODO: function that takes "sources"
#       -> need to change models loading
#       -> validate 'layer' key if present there
# FIXME: move validation step from normalization here for 'layer' & 'range' coexistence
# TODO: function that checks for 'range' or 'layer' at top level
# TODO: function that checks for 'layers' key at top level (the filter key)
# TODO: function that checks for base_model at source level

# TODO: we might have merge_method multiple times, for different ranges/slices
#       -> handle that by just validating all mentions of merge_method against known spec

# TODO: we might have models multiple times, for different ranges/slices
#       -> handle that by just validating all mentions of models against known spec

# TODO: do architecture validation for the models -> we need that in normalization
#       for the layer names

# --> FINALLY WE PASS ONLY SUBSET TO NORMALIZATION
# --> we create the model metadata + model? with the other subset -> deferring the loading
# --> until we have legality checked everything

    {
        "sources": [
            {"model": "model_1", "base_model": True},
            {"model": "model_2"}
        ],
        "range": [0, 2],
        "layers": ["mlp"],
        "merge_method": "slerp"
    },
