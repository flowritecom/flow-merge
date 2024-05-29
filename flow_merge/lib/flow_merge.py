from typing import Union, Dict, Any, Type
from pydantic import BaseModel
import yaml
from config import ApplicationConfig
from logger import Logger

class MergeConfig(BaseModel):
    param1: str
    param2: int
    device: str = "cpu"  # Default to 'cpu' if not set

class FlowMerge:
    def __init__(self, config: MergeConfig, logger=Logger().get_logger()):
        self.config = config
        self.logger = logger
        self.device = config.device

    def load(self):
        self.logger.info("Loading FlowMerge")
        print(f"Loading FlowMerge with config: {self.config}")

    def plan(self):
        self.logger.info("Planning FlowMerge")
        print(f"Planning FlowMerge with config: {self.config}")

    def run(self):
        self.logger.info("Running FlowMerge")
        print(f"Running FlowMerge with config: {self.config}")

    def eval(self):
        self.logger.info("Evaluating FlowMerge")
        print(f"Evaluating FlowMerge with config: {self.config}")

    def save(self):
        self.logger.info("Saving FlowMerge")
        print(f"Saving FlowMerge with config: {self.config}")

class ServiceContainer:
    def __init__(self, service_class: Type[FlowMerge], config: Union[Dict[str, Any], MergeConfig], logger, env: ApplicationConfig):
        if isinstance(config, dict):
            self.config = MergeConfig(**config)
        elif isinstance(config, MergeConfig):
            self.config = config
        else:
            raise TypeError("config must be a dictionary or a MergeConfig instance")

        self.service_instance = service_class(self.config, logger)
        self.env = env

    def get_service(self) -> FlowMerge:
        return self.service_instance

class DependencyInjector:
    def __init__(self):
        self.dependencies = {}

    def register(self, name: str, dependency: Any):
        self.dependencies[name] = dependency

    def inject(self, name: str) -> Any:
        return self.dependencies.get(name)

class ConfigLoader:
    def __init__(self, logger):
        self.logger = logger

    @classmethod
    def load(cls, config: Union[str, Dict[str, Any]], logger) -> Dict[str, Any]:
        logger.info("Loading configuration")
        if isinstance(config, str):
            return cls.from_yaml(config, logger)
        elif isinstance(config, dict):
            return cls.from_dict(config, logger)
        else:
            raise TypeError(
                "Input to load needs to be either a string path to a YAML config file or a dict"
            )

    @staticmethod
    def from_dict(data: Dict[str, Any], logger) -> Dict[str, Any]:
        logger.info("Loading from dict")
        return data

    @classmethod
    def from_yaml(cls, file_path: str, logger) -> Dict[str, Any]:
        logger.info(f"Loading from YAML file: {file_path}")
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

class MergeKitLoader(ConfigLoader):
    @classmethod
    def load(cls, config: Union[str, Dict[str, Any]], logger) -> Dict[str, Any]:
        logger.info("Loading MergeKit configuration")
        if isinstance(config, str):
            return cls.from_yaml(config, logger)
        elif isinstance(config, dict):
            return cls.from_dict(config, logger)
        else:
            raise TypeError(
                "Input to load needs to be either a string path to a YAML config file or a dict"
            )

class FlowMergeManager:
    def __init__(self, default_logger: Logger = None):
        self.containers = {}
        self.injector = DependencyInjector()
        self.default_logger = default_logger or Logger().get_logger()
        self.register_dependency('logger', self.default_logger)

    def register_service(self, name: str, service_class: Type[FlowMerge], config: Union[Dict[str, Any], MergeConfig], env: ApplicationConfig = None):
        logger = self.get_dependency('logger')
        env = env or ApplicationConfig()
        self.containers[name] = ServiceContainer(service_class, config, logger, env)
        service = self.containers[name].get_service()
        setattr(self, name, service)

        if hasattr(service, 'load'):
            self.load = service.load
        if hasattr(service, 'run'):
            self.run = service.run
        if hasattr(service, 'eval'):
            self.eval = service.eval

    def register_dependency(self, name: str, dependency: Any):
        self.injector.register(name, dependency)

    def get_dependency(self, name: str) -> Any:
        return self.injector.inject(name)






import unittest

class FlowMergeTest(unittest.TestCase):
    def test_service_registration_and_retrieval(self):
        merge = FlowMergeManager()
        logger = Logger().get_logger()
        merge.register_dependency('logger', logger)
        merge.register_service('merge', FlowMerge, {"param1": "value1", "param2": 2})

        self.assertTrue(hasattr(merge, 'containers'))
        self.assertEqual(merge.containers['merge'].get_service().config.param1, "value1")

    def test_dependency_injection(self):
        merge = FlowMergeManager()
        merge.register_dependency('db', 'DatabaseConnection')

        db = merge.get_dependency('db')
        self.assertEqual(db, 'DatabaseConnection')

    def test_loader_registration_and_usage(self):
        merge = FlowMergeManager()
        logger = Logger().get_logger()
        merge.register_dependency('logger', logger)
        merge.register_service('loader', ConfigLoader, {})
        merge.register_service('merge', FlowMerge, {"param1": "value1", "param2": 2})
        merge.load({"param1": "loaded_value1", "param2": 20})

        self.assertEqual(merge.containers['merge'].get_service().config.param1, "loaded_value1")
        self.assertEqual(merge.containers['merge'].get_service().config.param2, 20)

    def test_run_method(self):
        merge = FlowMergeManager()
        logger = Logger().get_logger()
        merge.register_dependency('logger', logger)
        merge.register_service('merge', FlowMerge, {"param1": "value1", "param2": 2})
        merge.run()

if __name__ == "__main__":
    unittest.main()
