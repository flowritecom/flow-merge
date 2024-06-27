from pprint import pprint
import yaml
import pytest
import unittest
from unittest.mock import patch
from io import StringIO

from flow_merge.lib.loaders.normalizer import NormalizationRunner, load_architecture


class TestNormalizationRunner(unittest.TestCase):
    def setUp(self):
        # default runner
        self.runner = NormalizationRunner("llama.json")

    # def test_different_range_at_sources_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 1]
    #         - model: B
    #         range: [1, 2]
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B', 'layer': 'model.layers.1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B', 'layer': 'model.layers.2'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_same_range_at_sources_level(self):
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     range: [0, 1]
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B', 'layer': 'model.layers.0'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B', 'layer': 'model.layers.1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_same_layer_at_sources_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         layer: model.layers.0.self_attn.q_proj.bias
    #         - model: B
    #         layer: model.layers.0.self_attn.q_proj.bias
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B', 'layer': 'model.layers.0.self_attn.q_proj.bias'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_different_layer_at_sources_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         layer: model.layers.0.self_attn.q_proj.bias
    #         - model: B
    #         layer: model.layers.1.self_attn.q_proj.bias
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B', 'layer': 'model.layers.1.self_attn.q_proj.bias'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_layer_at_top_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layer: model.layers.0.self_attn.q_proj.bias
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B', 'layer': 'model.layers.0.self_attn.q_proj.bias'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_layer_or_range_indicated_only_at_one_source():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         layer: model.layers.0.self_attn.q_proj.bias
    #         - model: B
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_slices_notation_exists():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     slices: ["slice1", "slice2"]
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'slices': ['slice1', 'slice2'],
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_no_slices_notation():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_no_base_model_designated():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_all_base_model_false_designated_illegal():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         is_base: false
    #         - model: B
    #         is_base: false
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'is_base': False},
    #                     {'model': 'B', 'is_base': False}
    #                 ],
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_both_base_model_false_and_true_designated_illegal():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     is_base: false
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'is_base': False,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_both_base_model_false_and_true_designated_at_one_source_illegal():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         is_base: true
    #         is_base: false
    #         - model: B
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'is_base': True, 'is_base': False},
    #                     {'model': 'B'}
    #                 ],
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_base_model_designated_at_top_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     base_model: "A"
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'base_model': 'A',
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_base_model_designated_at_top_level_and_sources_conflict():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         is_base: false
    #         - model: B
    #         is_base: true
    #     base_model: "A"
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'is_base': False},
    #                     {'model': 'B', 'is_base': True}
    #                 ],
    #                 'base_model': 'A',
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_layer_indicated_without_layer_filters():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         layer: model.layers.0.self_attn.q_proj.bias
    #         - model: B
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_layer_indicated_with_layer_filters():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         layer: model.layers.0.self_attn.q_proj.bias
    #         - model: B
    #         layer: model.layers.1.self_attn.q_proj.bias
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0.self_attn.q_proj.bias'},
    #                     {'model': 'B', 'layer': 'model.layers.1.self_attn.q_proj.bias'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_range_indicated_without_layer_filters():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 1]
    #         - model: B
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_range_indicated_without_layer_filters_2():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 1]
    #         - model: B
    #     is_base: true
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_template_indexing():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_non_existing_model_layer_in_layers_filter():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["non_existing_layer"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['non_existing_layer'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_duplicate_identical_slice_warn():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_duplicate_slice_warn():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_global_parameters_handling_case_1():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_merge_method_at_top_level_and_sources_level():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         merge_method: slerp
    #         - model: B
    #         merge_method: slerp
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: lerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'merge_method': 'slerp'},
    #                     {'model': 'B', 'merge_method': 'slerp'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'lerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_merge_method_is_string_and_belongs_to_set():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         - model: B
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A'},
    #                     {'model': 'B'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_range_outside_model_layers_available():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 100]
    #         - model: B
    #         range: [0, 100]
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B', 'layer': 'model.layers.0'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B', 'layer': 'model.layers.1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         # Add more slices as needed until range 100
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_range_is_non_negative():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [-1, 1]
    #         - model: B
    #         range: [-1, 1]
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.-1'},
    #                     {'model': 'B', 'layer': 'model.layers.-1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B', 'layer': 'model.layers.1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_range_needs_to_extend_at_least_one_layer():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 0]
    #         - model: B
    #         range: [0, 0]
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B', 'layer': 'model.layers.0'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected

    # def test_given_range_lengths_across_sources_mismatch():
    #     yaml_input = """
    #     slice:
    #     sources:
    #         - model: A
    #         range: [0, 1]
    #         - model: B
    #         range: [0, 2]
    #     is_base: true
    #     layers: ["mlp"]
    #     merge_method: slerp
    #     """
    #     expected = [
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.0'},
    #                     {'model': 'B', 'layer': 'model.layers.0'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'A', 'layer': 'model.layers.1'},
    #                     {'model': 'B', 'layer': 'model.layers.1'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         },
    #         {
    #             'slice': {
    #                 'sources': [
    #                     {'model': 'B', 'layer': 'model.layers.2'}
    #                 ],
    #                 'is_base': True,
    #                 'layers': ['mlp'],
    #                 'merge_method': 'slerp'
    #             }
    #         }
    #     ]
    #     actual = yaml.safe_load(yaml_input)
    #     assert actual == expected


if __name__ == "__main__":
    pytest.main()
