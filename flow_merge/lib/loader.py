from typing import Any, Dict, Union

import yaml


# these might get configs of their own
class ConfigLoader:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger

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
        return data

    def from_yaml(self, file_path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading from YAML file: {file_path}")
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
