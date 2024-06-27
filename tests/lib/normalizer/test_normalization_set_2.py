import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch


class TestNormalizationRunner(unittest.TestCase):
    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def setUp(self, mock_load_architecture):
        self.maxDiff = None
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        self.runner = NormalizationRunner("dummy_path.json")

    def test_range_without_layers_filter(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [0, 1]
            sources:
              - model: A
                base_model: True
              - model: B
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.mlp.weight", "model": "A", },
                        {"layer": "model.layers.0.mlp.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.mlp.weight", "model": "A", },
                        {"layer": "model.layers.1.mlp.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_range_with_layers_filter(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [0, 1]
            layers: ["self_attn"]
            sources:
              - model: A
                base_model: True
              - model: B
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.mlp.weight", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
            {
                "index": 2,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 3,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.mlp.weight", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_nontexisting_model_layer_in_layers_filter(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [0, 1]
            layers: ["xyz_not_existing"]
            sources:
              - model: A
                base_model: True
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)
        self.assertEqual("layer 'xyz_not_existing' does not exist in the model", e.exception.__str__())

    def test_range_outside_of_the_available_models_layers(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [1337, 1338]
            sources:
              - model: A
                base_model: True
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)
        self.assertEqual("provided layers range is outside of the source model layers range", e.exception.__str__())

    def test_invalid_range(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [2000, 1999]
            sources:
              - model: A
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("provided layers range is not positive", e.exception.__str__())

    def test_range_at_least_of_length_one(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            range: [2000, 2000]
            sources:
              - model: A
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("provided layers range is not positive", e.exception.__str__())

    def test_different_ranges_lengths_at_different_sources(self):
        yaml_input = """
        definition:
          - merge_method: slerp
            sources:
              - model: A
                range: [0, 1]
              - model: B
                range: [0, 999]
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual(
            "provided layers ranges are not the same length across all of the sources",
            e.exception.__str__()
        )
