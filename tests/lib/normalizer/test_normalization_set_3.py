import unittest

import yaml

from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch, MagicMock

from flow_merge.lib.model.architecture import ModelArchitectureProvider, ModelArchitecture, ModelWeight


class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model_arch_provider = MagicMock(ModelArchitectureProvider)
        arch = MagicMock(ModelArchitecture)
        arch.raw_weights = [
            ModelWeight(name="model.embed_tokens.weight", type="embed_tokens", layer_type="embedding"),
            ModelWeight(name="model.layers.{layer_index}.self_attn.k_proj.weight", type="self_attn",
                        layer_type="decoder"),
            ModelWeight(name="model.norm.weight", type="norm", layer_type="post_norm"),
            ModelWeight(name="lm_head.weight", type="lm_head", layer_type="head"),
        ]
        self.model_arch_provider.get_by_id.return_value = arch
        self.runner = NormalizationRunner(self.model_arch_provider)

    def test_special_layers_are_added_with_range_syntax(self):
        """
        Architecture defines special layer `lm_head` – it should be added in the output
        """
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
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "block_id": 1,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.1.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "block_id": None,
                "layer_type": "embedding",
                "output_layer_name": "model.embed_tokens.weight",
                "sources": [
                    {"is_base": True, "layer": "model.embed_tokens.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.embed_tokens.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "interpolate"
                }
            },
            {
                "block_id": None,
                "layer_type": "post_norm",
                "output_layer_name": "model.norm.weight",
                "sources": [
                    {"is_base": True, "layer": "model.norm.weight", "model": "A", "weight": 1.0},
                    {"layer": "model.norm.weight", "model": "B", "weight": 0.5},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "block_id": None,
                "layer_type": "head",
                "output_layer_name": "lm_head.weight",
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
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_lack_of_global_base_model(self):
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
            self.runner.normalize(yaml_loaded)

        self.assertEqual("Base model is missing", e.exception.__str__())

    def test_no_source_available_for_base(self):
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
            self.runner.normalize(yaml_loaded)

        self.assertEqual("No valid source found to set as base_model", e.exception.__str__())

    def test_slice_without_range_and_layer(self):
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
            self.runner.normalize(yaml_loaded)

        self.assertEqual("Slice provided without range of layers to merge", e.exception.__str__())
