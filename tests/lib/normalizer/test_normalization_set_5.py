import json
import os.path
import unittest
import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import MagicMock

from flow_merge.lib.model.architecture import ModelArchitecture, ModelArchitectureProvider, ModelWeight


class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model_arch_provider = MagicMock(ModelArchitectureProvider)
        arch = MagicMock(ModelArchitecture)
        arch.raw_weights = [
            ModelWeight(name="model.embed_tokens.weight", type="embed_tokens", layer_type="embedding"),
            ModelWeight(name="model.layers.{layer_index}.self_attn.k_proj", type="self_attn", layer_type="decoder"),
            ModelWeight(name="model.layers.{layer_index}.mlp.gate_proj.weight", type="mlp", layer_type="decoder"),
            ModelWeight(name="model.norm.weight", type="norm", layer_type="post_norm"),
        ]
        self.model_arch_provider.get_by_id.return_value = arch
        self.runner = NormalizationRunner(self.model_arch_provider)

    def test_range_syntax_supporting_single_number(self):
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                base_model: True
                range: 2
              - model: B
                range: 15
                weight: 0.69
        """
        expected_result_path = os.path.dirname(
            __file__) + "/expected_outputs/5_test_range_syntax_supporting_single_number.json"
        with open(expected_result_path, "r") as f:
            expected = json.load(f)

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)
