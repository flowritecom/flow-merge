from pprint import pprint
import unittest
from flow_merge.lib.flow_merge import ConfigLoader, FlowMergeManager
from flow_merge.lib.logger import Logger
from flow_merge.lib.config import ApplicationConfig

class FlowMergeTest(unittest.TestCase):

    def test_service_class_registering(self):
        merge = FlowMergeManager()
        logger = Logger().get_logger()
        merge.register_dependency("logger", logger)
        config = ApplicationConfig()
        merge.register_service("loader", ConfigLoader, config)
        self.assertTrue(hasattr(merge, "containers"))
        what = merge.containers["loader"].get_service()

        pprint(what)
        self.assertTrue(hasattr(
            merge.containers["loader"].get_service().env, "hf_token"))


    def test_dependency_injection(self):
        merge = FlowMergeManager()
        merge.register_dependency("db", "DatabaseConnection")
        db = merge.get_dependency("db")
        self.assertEqual(db, "DatabaseConnection")

    def test_load_method(self):
        merge = FlowMergeManager()
        logger = Logger().get_logger()
        merge.register_dependency("logger", logger)
        config = ApplicationConfig()
        merge.register_service("loader", ConfigLoader, config)
        self.assertTrue(hasattr(merge, 'load'))


if __name__ == "__main__":
    unittest.main()
