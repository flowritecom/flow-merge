from typing import Any

from flow_merge.lib.extend import DependencyInjector, ServiceContainer
from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger


class FlowMerge:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load(self):
        self.logger.info("Loading")

    def plan(self):
        self.logger.info("Planning")

    def run(self):
        self.logger.info("Running")

    def save(self):
        self.logger.info("Saving")


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
