import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch, mock_open
import yaml
from flow_merge.lib.loaders.loader import ConfigLoader

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.env = MagicMock()
        self.logger = MagicMock()
        self.validation_runner = MagicMock()
        self.loader = ConfigLoader(env=self.env, logger=self.logger, validation_runner=self.validation_runner)

    def test_validate_success(self):
        raw_data = {"key": "value"}
        self.validation_runner.return_value = raw_data

        result = self.loader.validate(raw_data)

        self.logger.info.assert_called_with("Validating configuration")
        self.validation_runner.assert_called_with(**raw_data, env=self.env, logger=self.logger)
        self.assertEqual(result, raw_data)

    def test_validate_failure(self):
        raw_data = {"key": "value"}
        self.validation_runner.side_effect = ValueError("Invalid config")

        # Capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        result = self.loader.validate(raw_data)

        # Reset redirect.
        sys.stdout = sys.__stdout__

        self.logger.info.assert_called_with("Validating configuration")
        self.assertIn("Validation error: Invalid config", captured_output.getvalue())
        self.assertIsNone(result)

    def test_load_from_dict(self):
        config = {"key": "value"}
        self.loader.validate = MagicMock(return_value=config)

        result = self.loader.load(config)

        self.loader.validate.assert_called_with(config)
        self.assertEqual(result, config)

    def test_load_from_yaml(self):
        config = {"key": "value"}
        yaml_content = yaml.dump(config)
        self.loader.validate = MagicMock(return_value=config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            result = self.loader.load('config.yaml')

        self.loader.validate.assert_called_with(config)
        self.assertEqual(result, config)

    def test_load_invalid_type(self):
        with self.assertRaises(TypeError):
            self.loader.load(123)

    def test_from_dict(self):
        data = {"key": "value"}
        self.loader.validate = MagicMock(return_value=data)

        result = self.loader.from_dict(data)

        self.logger.info.assert_called_with("Loading from dict")
        self.loader.validate.assert_called_with(data)
        self.assertEqual(result, data)

    def test_from_yaml(self):
        config = {"key": "value"}
        yaml_content = yaml.dump(config)
        self.loader.validate = MagicMock(return_value=config)

        with patch('builtins.open', mock_open(read_data=yaml_content)):
            result = self.loader.from_yaml('config.yaml')

        self.logger.info.assert_called_with("Loading from YAML file: config.yaml")
        self.loader.validate.assert_called_with(config)
        self.assertEqual(result, config)

if __name__ == '__main__':
    unittest.main()
