import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

# import pydevd_pycharm
# pydevd_pycharm.settrace('172.17.0.1', port=9898, stdoutToServer=True, stderrToServer=True)

class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.runner = NormalizationRunner()

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_special_layers_are_added_with_range_syntax(self, mock_load_architecture):
        """
        Architecture defines special layer `lm_head` – it should be added as a last slice in the output
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.embed_tokens.weight", "type": "embed"},
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.norm.weight", "type": "norm"},
                {"name": "lm_head.weight", "type": "lm_head"},
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
                        {"base_model": True, "layer": "model.embed_tokens.weight", "model": "A", },
                        {"layer": "model.embed_tokens.weight", "model": "B", },
                    ],
                    "merge_method": "interpolate"
                },
            },
            {
                "output_layer_id": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "output_layer_id": 2,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "output_layer_id": 3,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.norm.weight", "model": "A", },
                        {"layer": "model.norm.weight", "model": "B", },
                    ],
                    "merge_method": "interpolate"
                },
            },
            {
                "output_layer_id": 4,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "lm_head.weight", "model": "A", },
                        {"layer": "lm_head.weight", "model": "B", },
                    ],
                    "merge_method": "interpolate"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_lack_of_global_base_model(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        yaml_input = """
            definition:
              - merge_method: slerp
                sources:
                  - model: A
                    base_model: True
                    range: [0, 1]
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("Base model is missing", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_no_source_available_for_base(self, mock_load_architecture):
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
                    base_model: False
                    range: [0, 1]
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("No valid source found to set as base_model", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_slice_without_range_and_layer(self, mock_load_architecture):
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
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("Neither range or layers defined for merging", e.exception.__str__())

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_not_mergable_layer_used(self, mock_load_architecture):
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
                    layer: model.lm_head
            """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("Layer defined for merging must be a hidden layer (pattern layer)", e.exception.__str__())
