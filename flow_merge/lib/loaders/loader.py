from typing import Any, Callable, Dict, Union

import yaml

from flow_merge.lib.validators.runner import ValidationRunner


# these might get configs of their own
# FIXME: takes a normalizer ?
class ConfigLoader:
    def __init__(self, env, logger, validation_runner: Callable = ValidationRunner):
        self.env = env
        self.logger = logger
        self.validation_runner = validation_runner

    def validate(self, raw_data: dict):
        self.logger.info("Validating configuration")
        try:
            validated_data = self.validation_runner(
                **raw_data, env=self.env, logger=self.logger
            )
            print(validated_data)
            return validated_data
        except ValueError as e:
            print(f"Validation error: {e}")

    def load(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info("Loading configuration")
        if isinstance(config, str):
            return self.from_yaml(config)
        elif isinstance(config, dict):
            return self.from_dict(config)
        else:
            raise TypeError(
                "Input to load needs to be either a string path to a YAML config file or a dict"
            )

    def from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Loading from dict")
        validated_data = self.validate(data)
        return validated_data

    def from_yaml(self, file_path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading from YAML file: {file_path}")
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return self.validate(data)
