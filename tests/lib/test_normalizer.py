# test_normalizer.py
import unittest
from unittest.mock import patch
from flow_merge.lib.loaders.normalizer import NormalizationRunner, load_architecture


class TestNormalizationRunner(unittest.TestCase):
    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def setUp(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.norm.weight", "type": "post_norm"},
                {"name": "model.embed_tokens.weight", "type": "embed"},
                {"name": "model.lm_head.weight", "type": "lm_head"}
            ]
        }
        self.runner = NormalizationRunner("dummy_path.json")

    def test_load_architecture(self):
        with patch('flow_merge.lib.loaders.normalizer.pkg_resources.resource_string', return_value=b'{}') as mock_resource_string:
            result = load_architecture("dummy_path.json")
            mock_resource_string.assert_called_once()
            self.assertEqual(result, {})

    def test_ensure_base_model(self):
        slice = {
            "sources": [
                {"model": "model_1", "base_model": False},
                {"model": "model_2"}
            ]
        }
        result = self.runner._ensure_base_model(slice)
        self.assertTrue(any(src["base_model"] for src in result["sources"]))

    def test_create_slice(self):
        sources = [{"model": "model_1"}]
        layer = "model.layers.0.self_attn.k_proj.weight"
        merge_method = "slerp"
        result = self.runner._create_slice(sources, layer, merge_method)
        self.assertEqual(result["slice"]["merge_method"], merge_method)
        self.assertEqual(result["slice"]["sources"][0]["layer"], layer)

    def test_process_template_slices_with_range(self):
        slice = {
            "sources": [{"model": "model_1"}],
            "range": [0, 1],
            "merge_method": "slerp"
        }
        result = self.runner._process_template_slices(slice)
        self.assertEqual(len(result), 2)

    def test_process_template_slices_with_layer(self):
        slice = {
            "sources": [{"model": "model_1"}],
            "layer": "model.layers.0.self_attn.k_proj.weight",
            "merge_method": "slerp"
        }
        result = self.runner._process_template_slices(slice)
        self.assertEqual(len(result), 1)

    def test_process_special_layers(self):
        normalized_data = [
            {"slice": {"sources": [{"model": "model_1", "layer": "model.layers.0.self_attn.k_proj.weight"}], "merge_method": "slerp"}}
        ]
        result = self.runner._process_special_layers(normalized_data)

        self.assertEqual(len(result), 3)

    def test_move_embed_slice_to_top(self):
        normalized_data = [
            {"slice": {"sources": [{"model": "model_1", "layer": "model.layers.0.self_attn.k_proj.weight"}], "merge_method": "slerp"}},
            {"slice": {"sources": [{"model": "model_1", "layer": "model.embed_tokens.weight"}], "merge_method": "slerp"}}
        ]
        result = self.runner._move_embed_slice_to_top(normalized_data)
        self.assertIn("embed_tokens", result[0]["slice"]["sources"][0]["layer"])

    def test_add_indices_to_slices(self):
        normalized_data = [
            {"slice": {"sources": [{"model": "model_1", "layer": "model.layers.0.self_attn.k_proj.weight"}], "merge_method": "slerp"}}
        ]
        result = self.runner._add_indices_to_slices(normalized_data)
        self.assertIn("index", result[0])
        self.assertEqual(result[0]["index"], 0)

    def test_validate_slice(self):
        with self.assertRaises(ValueError):
            self.runner._validate_slice({"range": [0, 1], "layer": "model.layers.0.self_attn.k_proj.weight"})

        with self.assertRaises(ValueError):
            self.runner._validate_slice({"sources": [{"model": "model_1", "range": [0, 1]}]})

        with self.assertRaises(ValueError):
            self.runner._validate_slice({"range": [0, 1], "sources": [{"model": "model_1", "layer": "model.layers.0.self_attn.k_proj.weight"}]})

    def test_normalize(self):
        slices = [
            {
                "sources": [
                    {"model": "model_1", "base_model": True},
                    {"model": "model_2"}
                ],
                "range": [0, 2],
                "layers": ["mlp"],
                "merge_method": "slerp"
            },
            {
                "sources": [
                    {"model": "model_1"},
                    {"model": "model_2",}
                ],
                "range": [2, 3],
                "merge_method": "slerp"
            }
        ]
        result = self.runner.normalize(slices)
        # Check if the result has the expected length, which should be the number of expanded slices plus special layers
        self.assertEqual(len(result), 8)
        # Check the presence of special layers in the result
        self.assertTrue(any('embed_tokens' in entry['slice']['sources'][0]['layer'] for entry in result))
        self.assertTrue(any('lm_head' in entry['slice']['sources'][0]['layer'] for entry in result))
        self.assertTrue(any('norm' in entry['slice']['sources'][0]['layer'] for entry in result))

if __name__ == '__main__':
    unittest.main()
