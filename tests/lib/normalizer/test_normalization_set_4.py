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
            ModelWeight(name="model.embed_tokens.weight", type="embed_tokens", layer_type="embedding", ),
            ModelWeight(name="model.layers.{layer_index}.self_attn.k_proj", type="self_attn", layer_type="decoder"),
            ModelWeight(name="model.layers.{layer_index}.mlp.gate_proj.weight", type="mlp", layer_type="decoder"),
            ModelWeight(name="model.norm.weight", type="norm", layer_type="post_norm", ),
        ]
        self.model_arch_provider.get_by_id.return_value = arch
        self.runner = NormalizationRunner(self.model_arch_provider)

    def test_output_layer_indexing_correct_for_range_syntax_(self):
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                base_model: True
                range: [12, 13]
              - model: B
                range: [77, 78]
        """
        expected_result_path = os.path.dirname(
            __file__) + "/expected_outputs/4_output_layer_indexing_correct_for_range_syntax.json"
        with open(expected_result_path, "r") as f:
            expected = json.load(f)

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_output_layer_indexing_correct_for_range_syntax_multiple_slices(self):
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                base_model: True
                range: [12, 13]
              - model: B
                range: [77, 78]
          - merge_method: 
              name: slerp
            sources:
              - model: X
                base_model: True
                range: [69, 71]
              - model: Y
                range: [33, 35]
          - merge_method: 
              name: slerp
            layers: ["self_attn"]       # One with filter
            sources:
              - model: X
                base_model: True
                range: [123, 125]
              - model: Y
                range: [99, 101]
        """
        expected_result_path = os.path.dirname(
            __file__) + "/expected_outputs/4_output_layer_indexing_correct_for_range_syntax_multiple.json"
        with open(expected_result_path, "r") as f:
            expected = json.load(f)

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertCountEqual(expected, processed)

    def test_output_layer_indexing_correct_for_range_syntax_multiple_slices_different_order(self):
        yaml_input = """
        base_model: A
        definition:
          - merge_method: 
              name: slerp
            sources:
              - model: A
                base_model: True
                range: [12, 13]
              - model: B
                range: [77, 78]
          - merge_method: 
              name: slerp
            layers: ["self_attn"]       # One with filter
            sources:
              - model: X
                base_model: True
                range: [123, 125]
              - model: Y
                range: [99, 101]
          - merge_method: 
              name: slerp
            sources:
              - model: X
                base_model: True
                range: [69, 71]
              - model: Y
                range: [33, 35]
        """
        expected_result_path = os.path.dirname(
            __file__) + "/expected_outputs/4_output_layer_indexing_correct_for_range_syntax_multiple_diff_order.json"
        with open(expected_result_path, "r") as f:
            expected = json.load(f)

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertCountEqual(expected, processed)

    def test_output_layer_indexing_correct_for_range_syntax_exotic(self):
        yaml_input = """
        name: test_arithmetic
        base_model: Qwen/Qwen2-0.5B
        definition:
          - sources:
            - range: [0, 18]
              model: Qwen/Qwen2-0.5B
              is_base: True
              # weight: 0.4
            - range: [0, 18]
              model: Qwen/Qwen2-0.5B-Chat
              # weight: 0.3
            merge_method:
              name: addition-task-arithmetic
              params:
                scaling-coefficient: 0.8
                normalize: True
          - sources:
            - range: [22,23]
              model: Qwen/Qwen2-0.5B-Chat
            merge_method:
              name: passthrough

            """
        expected_result_path = os.path.dirname(
            __file__) + "/expected_outputs/4_output_layer_indexing_correct_for_range_syntax_exotic.json"
        with open(expected_result_path, "r") as f:
            expected = json.load(f)

        yaml_loaded = yaml.safe_load(yaml_input)
        processed = self.runner.normalize(yaml_loaded)
        self.assertCountEqual(expected, processed)
