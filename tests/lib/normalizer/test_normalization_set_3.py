import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch

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
                {"name": "lm_head.weight", "type": "lm_head"},
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
