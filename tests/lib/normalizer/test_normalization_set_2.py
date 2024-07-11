import unittest

import yaml

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

from flow_merge.lib.validators import DirectorySettings


class TestNormalizationRunner(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_range_without_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn", "layer_type": "decoder"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp", "layer_type": "decoder"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method:
              name: slerp
            sources:
              - model: A
                is_base: True
                range: [0, 1]
                weight: 0.5
              - model: B
                range: [0, 1]
                weight: 1.0
        """
        expected = [
            {
                "output_layer_id": 0,
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp",
                }
            },
            {
                "output_layer_id": 0,
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.mlp.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.mlp.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 1,
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 1,
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.mlp.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.1.mlp.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        normalizer = NormalizationRunner(ApplicationConfig(), None)
        processed, num_hidden_layers = normalizer.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual(2, num_hidden_layers)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_range_with_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn", "layer_type": "decoder"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp", "layer_type": "decoder"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            layers: ["attn"]
            sources:
              - model: A
                is_base: True
                range: [0, 1]
                weight: 0.5
              - model: B
                range: [0, 1]
                weight: 1.0
        """
        expected = [
            {
                "output_layer_id": 0,
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 1,
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 0,
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.mlp.weight", "model": "A"},
                ],
                "merge_method": {
                    "name": "passthrough"
                }
            },
            {
                "output_layer_id": 1,
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.mlp.weight", "model": "A"},
                ],
                "merge_method": {
                    "name": "passthrough"
                }
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        normalizer = NormalizationRunner(ApplicationConfig(), None)
        processed, num_hidden_layers = normalizer.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual(2, num_hidden_layers)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_nontexisting_model_layer_in_layers_filter(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn", "layer_type": "decoder"},
            ]
        }
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            layers: ["xyz_not_existing"]
            sources:
              - model: A
                is_base: True
                range: [0, 1]
                weight: 0.5
              - model: B
                range: [0, 1]
                weight: 1.0
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        normalizer = NormalizationRunner(ApplicationConfig(), None)

        with self.assertRaises(Exception) as e:
            normalizer.normalize(yaml_loaded, directory_settings=DirectorySettings())
        self.assertEqual("Layer 'xyz_not_existing' does not exist in the model", e.exception.__str__())
