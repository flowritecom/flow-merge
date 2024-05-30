import asyncio
from typing import Any, Dict, Union

import yaml

from flow_merge.lib.loaders.loader import ConfigLoader
from flow_merge.lib.validators.runner import async_runner


# these might get configs of their own
class AsyncConfigLoader(ConfigLoader):
    def __init__(self, env, logger):
        super().__init__(env, logger, validation_runner=async_runner)

    async def validate(self, raw_data: dict):
        self.logger.info("Validating configuration")
        try:
            validated_data = await self.validation_runner(raw_data)
            print(validated_data)
            return validated_data
        except ValueError as e:
            print(f"Validation error: {e}")

    async def load(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info("Loading configuration")
        if isinstance(config, str):
            return await self.from_yaml(config)
        elif isinstance(config, dict):
            return await self.from_dict(config)
        else:
            raise TypeError(
                "Input to load needs to be either a string path to a YAML config file or a dict"
            )

    async def from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Loading from dict")
        validated_data = await self.validate(data)
        return validated_data

    async def from_yaml(self, file_path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading from YAML file: {file_path}")
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return await self.validate(data)
