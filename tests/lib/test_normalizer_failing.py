# test_normalizer_additional.py
import unittest
from unittest.mock import patch
from flow_merge.lib.loaders.normalizer import NormalizationRunner


class TestNormalizationRunner(unittest.TestCase):
    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def setUp(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
                {"name": "model.norm.weight", "type": "post_norm"},
                {"name": "model.embed_tokens.weight", "type": "embed"},
                {"name": "model.lm_head.weight", "type": "lm_head"}
            ]
        }
        self.runner = NormalizationRunner("dummy_path.json")

    def test_normalize(self):
        slices = [
            {
                "sources": [
                    {"model": "model_1", "base_model": True},
                    {"model": "model_2"}
                ],
                "range": [0, 0],
                "layers": ["mlp"],
                "merge_method": "slerp"
            },
        ]
        result = self.runner.normalize(slices)

        # Slice which is the actual merge should have the method as defined in the user config
        self.assertEqual(result[2]["slice"]["merge_method"], "slerp")

        # When layers filter is applied, the slices creating the output block should have the same index
        self.assertEqual(result[1]["index"], result[2]["index"])

    def test_normalize_two(self):
        slices = [
            {
                "sources": [
                    {"model": "model_1", "base_model": True},
                    {"model": "model_2"}
                ],
                "range": [0, 0],
                "layers": ["mlp"],
                "merge_method": "slerp"
            },
            {
                "sources": [
                    {"model": "model_1"},
                    {"model": "model_2", }
                ],
                "range": [2, 3],
                "merge_method": "slerpX"
            }
        ]
        result = self.runner.normalize(slices)

        # First slice is normalized to two slices because of layer filter, they should have the same index
        self.assertEqual(result[1]["index"], result[2]["index"])

        # Second slice normalized to two slices (one per block layer) – should have the same index
        self.assertEqual(result[3]["index"], result[4]["index"])

        # Last two slices (special ones) should have the same merge method as the last slice from user config?
        # Somehow it's persisted. Probably because objects are not deep copied but instead python uses references?
        self.assertNotEqual("slerpX", result[7]["slice"]["merge_method"])
        self.assertNotEqual("slerpX", result[8]["slice"]["merge_method"])


if __name__ == '__main__':
    unittest.main()
