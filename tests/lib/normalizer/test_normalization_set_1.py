import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.runner = NormalizationRunner()

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_same_range_at_sources_level(self, mock_load_architecture):
        """
        Valid configuration – same length `range` in both sources
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: slerp
            sources:
              - model: A
                base_model: True
                range: [0, 1]
              - model: B
                range: [0, 1]
        """
        expected = [
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "output_layer_id": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_different_range_at_sources_level(self, mock_load_architecture):
        """
        Valid configuration – different `range` values in both sources, but still the same length in both.
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: slerp
            sources:
              - model: A
                base_model: True
                range: [0, 1]
              - model: B
                range: [5, 6]
        """
        expected = [
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.5.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "output_layer_id": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.6.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_no_slices_defined(self):
        """
        Illegal – no slices defined in the config
        """
        with self.assertRaises(Exception, msg="at least one slice configuration must be provided"):
            self.runner.normalize([])

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_no_base_model_defined(self, mock_load_architecture):
        """
        No base model defined, neither at top level or in sources.
        In that scenario, take the first model as a base one.
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: slerp
            sources:
              - model: A
                layer: "model.layers.0.self_attn.k_proj.weight"
              - model: B
                layer: "model.layers.0.self_attn.k_proj.weight"
        """
        expected = [
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_layer_defined(self, mock_load_architecture):
        """
        Specific layer names defined at source level.
        The output should fill in the other layers (mlp in this case) in the same output block,
        indicated by `index` value (should be `0` in both output slices).
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: slerp
            sources:
              - model: A
                layer: "model.layers.12.self_attn.k_proj.weight"
              - model: B
                layer: "model.layers.12.self_attn.k_proj.weight"
        """
        expected = [
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.12.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.mlp.weight", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        runner = NormalizationRunner()
        processed = runner.normalize(yaml_loaded)

        self.assertEqual(2, len(processed))
        self.assertEqual(expected, processed)
