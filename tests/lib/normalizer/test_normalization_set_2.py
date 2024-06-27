import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

class TestNormalizationRunner(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_range_without_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
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
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.mlp.weight", "model": "A", },
                        {"layer": "model.layers.0.mlp.weight", "model": "B", },
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
            {
                "output_layer_id": 1,
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
        normalizer = NormalizationRunner()
        processed = normalizer.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_range_with_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
        definition:
          - merge_method: slerp
            layers: ["attn"]
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
            {
                "output_layer_id": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.mlp.weight", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
            {
                "output_layer_id": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.mlp.weight", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        normalizer = NormalizationRunner()
        processed = normalizer.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_nontexisting_model_layer_in_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        yaml_input = """
        definition:
          - merge_method: slerp
            layers: ["xyz_not_existing"]
            sources:
              - model: A
                base_model: True
                range: [0, 1]
              - model: B
                range: [0, 1]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        normalizer = NormalizationRunner()

        with self.assertRaises(Exception) as e:
            normalizer.normalize(yaml_loaded)
        self.assertEqual("Layer 'xyz_not_existing' does not exist in the model", e.exception.__str__())
