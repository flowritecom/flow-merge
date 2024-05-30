from typing import Any


class ServiceContainer:
    def __init__(self, service_class, env=None, logger=None):
        self.service_instance = service_class(env=env, logger=logger)
        self.env = env
        self.logger = logger

    def get_service(self):
        return self.service_instance


class DependencyInjector:
    def __init__(self):
        self.dependencies = {}

    def register(self, name: str, dependency: Any):
        self.dependencies[name] = dependency

    def inject(self, name: str) -> Any:
        return self.dependencies.get(name)
