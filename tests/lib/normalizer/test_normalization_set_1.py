import unittest

import yaml

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import MagicMock

from flow_merge.lib.model.architecture import ModelArchitectureProvider, ModelArchitecture, ModelWeight


class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model_arch_provider = MagicMock(ModelArchitectureProvider)
        arch = MagicMock(ModelArchitecture)
        arch.raw_weights = [
            ModelWeight(name="model.layers.{layer_index}.self_attn.k_proj.weight", type="self_attn",
                        layer_type="decoder"),
        ]
        self.model_arch_provider.get_by_id.return_value = arch
        self.runner = NormalizationRunner(self.model_arch_provider)

    def test_same_range_at_sources_level(self):
        """
        Valid configuration – same length `range` in both sources
        """
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                base_model: True
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
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_different_range_at_sources_level(self):
        """
        Valid configuration – different `range` values in both sources, but still the same length in both.
        """
        yaml_input = """
        base_model: A
        definition:
          - merge_method:
              name: slerp
            sources:
              - model: A
                base_model: True
                range: [0, 1]
                weight: 0.5
              - model: B
                range: [5, 6]
                weight: 1.0
        """
        expected = [
            {
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.5.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
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
                    {"layer": "model.layers.6.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
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
            self.runner.normalize({})

    def test_no_base_model_defined(self):
        """
        No base model defined, neither at top level or in sources.
        In that scenario, take the first model as a base one.
        """
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                range: [0, 1]
                weight: 0.5
              - model: B
                range: [3, 4]
                weight: 1.0
        """
        expected = [
            {
                "block_id": 0,
                "layer_type": "decoder",
                "output_layer_name": "model.layers.0.self_attn.k_proj.weight",
                "sources": [
                    {"is_base": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", "weight": 0.5},
                    {"layer": "model.layers.3.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
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
                    {"layer": "model.layers.4.self_attn.k_proj.weight", "model": "B", "weight": 1.0},
                ],
                "merge_method": {
                    "name": "slerp"
                }
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

