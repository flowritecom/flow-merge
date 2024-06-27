from flow_merge.lib.model.metadata.model_metadata import ModelMetadata
from flow_merge.lib.model.metadata.file_validators._adapter import has_adapter_files
from flow_merge.lib.model.metadata.file_validators._integrity import has_config_json
from flow_merge.lib.model.metadata.file_validators._tokenizer import has_tokenizer_config, has_tokenizer_file
from flow_merge.lib.model.metadata.file_validators._safetensors import has_safetensors_files, has_safetensors_index
from flow_merge.lib.model.metadata.file_validators._pytorch_bin import has_pytorch_bin_files, has_pytorch_bin_index

# (attribute effected, test function, missing message)
integrity_checks = [
    ("has_config", has_config_json, "Missing config.json file"),
]

tokenizer_checks = [
    ("has_tokenizer_config", has_tokenizer_config, "Missing tokenizer_config.json file"),
    ("has_vocab", has_tokenizer_file, "Missing tokenizer vocabulary file"),
]

safetensor_checks = [
    ("has_safetensor_files", has_safetensors_files, "Missing .safetensors files"),
    ("has_safetensors_index", has_safetensors_index, "Missing model.safetensors.index.json file"),
]

pytorch_bin_checks = [
    ("has_pytorch_bin_files", has_pytorch_bin_files, "Missing pytorch_model .bin files"),
    ("has_pytorch_bin_index", has_pytorch_bin_index, "Missing pytorch_model.bin.index.json file"),
]

adapter_checks = [
    ("has_adapter", has_adapter_files, "Missing adapter files"),
]

class FileListValidator:
    def __init__(self, env, logger) -> None:
        self.env = env
        self.logger = logger
        self.checks = [
            integrity_checks,
            tokenizer_checks,
            safetensor_checks,
            pytorch_bin_checks,
            adapter_checks
        ]

    def check(self, metadata: ModelMetadata):
        if metadata.file_metadata_list:
            file_list = [
                file_metadata.filename for file_metadata in metadata.file_metadata_list
            ]

        for check in self.checks:
            for test_set in check:
                attribute_name, test_func, missing_message = test_set
                tested_outcome = test_func(file_list)

                if tested_outcome is False:
                    self.logger.info(missing_message)

                setattr(metadata, attribute_name, tested_outcome)
        
        return metadata
                
    