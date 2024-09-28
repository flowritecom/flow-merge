import unittest
from unittest.mock import MagicMock

import yaml

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from flow_merge.lib.model.architecture import ModelArchitecture, ModelWeight, ModelArchitectureProvider


class TestNormalizationRunner(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self.model_arch_provider = MagicMock(ModelArchitectureProvider)
        arch = MagicMock(ModelArchitecture)
        arch.raw_weights = [
            ModelWeight(name="model.layers.{layer_index}.self_attn.k_proj.weight", type="self_attn", layer_type="decoder"),
            ModelWeight(name="model.layers.{layer_index}.mlp.weight", type="mlp", layer_type="decoder"),
        ]
        self.model_arch_provider.get_by_id.return_value = arch
        self.runner = NormalizationRunner(self.model_arch_provider)

    def test_range_without_layers_filter(self):
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
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp",
                }
            },
            {
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.mlp.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.mlp.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.mlp.weight", "model": "B", "weight": 1.0},
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
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "block_id": 1,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.1.mlp.weight",
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
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_range_with_layers_filter(self):
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            layers: ["self_attn"]
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
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
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
                    {"is_base": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
            {
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.mlp.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.mlp.weight", "model": "A", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "passthrough"
                }
            },
            {
                "block_id": 1,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.1.mlp.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.1.mlp.weight", "model": "A", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "passthrough"
                }
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)


    def test_nontexisting_model_layer_in_layers_filter(self):
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

        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)
        self.assertEqual("Layer 'xyz_not_existing' does not exist in the model", e.exception.__str__())
