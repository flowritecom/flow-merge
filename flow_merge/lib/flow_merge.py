from typing import Union, Dict, Any
import yaml
from .config import ApplicationConfig
from .logger import Logger


class FlowMerge:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load(self):
        self.logger.info("Loading FlowMerge")
        print(f"Loading FlowMerge with config: {self.config}")

    def plan(self):
        self.logger.info("Planning FlowMerge")
        print(f"Planning FlowMerge with config: {self.config}")

    def run(self):
        self.logger.info("Running FlowMerge")
        print(f"Running FlowMerge with config: {self.config}")

    def save(self):
        self.logger.info("Saving FlowMerge")
        print(f"Saving FlowMerge with config: {self.config}")


class ServiceContainer:
    def __init__(
        self, service_class, env=None, logger=None
    ):
        self.service_instance = service_class(env=env, logger=logger)
        self.env = env
        self.logger = logger

    def get_service(self) -> FlowMerge:
        return self.service_instance


class DependencyInjector:
    def __init__(self):
        self.dependencies = {}

    def register(self, name: str, dependency: Any):
        self.dependencies[name] = dependency

    def inject(self, name: str) -> Any:
        return self.dependencies.get(name)


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


class FlowMergeManager:
    def __init__(self, default_logger: Logger = None):
        self.containers = {}
        self.injector = DependencyInjector()
        self.default_logger = default_logger or Logger().get_logger()
        self.register_dependency("logger", self.default_logger)

    def register_service(
        self,
        name: str,
        service_class,
        env: ApplicationConfig = None,
    ):
        # logger registered externally?
        logger = self.get_dependency("logger")

        env = env or ApplicationConfig()
        self.containers[name] = ServiceContainer(service_class, env, logger)
        service = self.containers[name].get_service()
        setattr(self, name, service)

        if hasattr(service, "load"):
            self.load = service.load
        if hasattr(service, "run"):
            self.run = service.run
        if hasattr(service, "eval"):
            self.eval = service.eval

    def register_dependency(self, name: str, dependency: Any):
        self.injector.register(name, dependency)

    def get_dependency(self, name: str) -> Any:
        return self.injector.inject(name)
