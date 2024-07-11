import unittest

import yaml

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

from flow_merge.lib.validators import DirectorySettings


class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.runner = NormalizationRunner(ApplicationConfig(), None)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_special_layers_are_added_with_range_syntax(self, mock_load_architecture):
        """
        Architecture defines special layer `lm_head` – it should be added as a last slice in the output
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.embed_tokens.weight", "type": "embed", "layer_type": "embedding"},
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn", "layer_type": "decoder"},
                {"name": "model.norm.weight", "type": "norm", "layer_type": "post_norm"},
                {"name": "lm_head.weight", "type": "lm_head", "layer_type": "head"},
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
                weight: 1.0
              - model: B
                range: [0, 1]
                weight: 0.5
        """
        expected = [
            {
                "output_layer_id": 0,
                "sources": [
                    {"is_base": True, "layer": "model.embed_tokens.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.embed_tokens.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "interpolate"
                }
            },
            {
                "output_layer_id": 1,
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 2,
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "output_layer_id": 3,
                "sources": [
                    {"is_base": True, "layer": "model.norm.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.norm.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "interpolate"
                }
            },
            {
                "output_layer_id": 4,
                "sources": [
                    {"is_base": True, "layer": "lm_head.weight", "model": "A", "weight": 1.0},
                    {"layer": "lm_head.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "interpolate"
                }
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed, num_hidden_layers = self.runner.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual(2, num_hidden_layers)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_lack_of_global_base_model(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn", "layer_type": "decoder"},
            ]
        }
        yaml_input = """
            definition:
              - merge_method: 
                  name: slerp
                sources:
                  - model: A
                    is_base: True
                    range: [0, 1]
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual("Base model is missing", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_no_source_available_for_base(self, mock_load_architecture):
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
                sources:
                  - model: A
                    is_base: False
                    range: [0, 1]
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual("No valid source found to set as base_model", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_slice_without_range_and_layer(self, mock_load_architecture):
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
                sources:
                  - model: A
                    is_base: True
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual("Neither range or layers defined for merging", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_not_mergable_layer_used(self, mock_load_architecture):
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
                sources:
                  - model: A
                    is_base: True
                    layer: model.lm_head
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded, directory_settings=DirectorySettings())

        self.assertEqual("Layer defined for merging must be a hidden layer (pattern layer)", e.exception.__str__())
