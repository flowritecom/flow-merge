import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch
from pprint import pprint

from flow_merge.lib.validators.slice_validator import SliceValidator
import pydevd_pycharm

pydevd_pycharm.settrace('172.17.0.1', port=9898, stdoutToServer=True, stderrToServer=True)


class TestSliceValidator(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_all_models_not_base(self):
        """
        All models marked as not base, exception
        """

        yaml_input = """
        definition:
          - merge_method: slerp
            range: [0, 1]
            sources:
              - model: A
                base_model: False
              - model: B
                base_model: False
        """

        yaml_loaded = yaml.safe_load(yaml_input)

        validator = SliceValidator()
        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("No valid source found to set as base_model", e.exception.__str__())

    def test_range_indicated_only_at_once_source(self):
        """
        Illegal  – `range` syntax applied to one source only. If applied, must be applied
        at top level or to all sources (and the length must be the same for all `range`s).
        """
        yaml_input = """
        definition:
          - merge_method: slerp
            sources:
              - model: A
                base_model: True
                range: [0, 1] 
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])
        self.assertEqual("If used, `range` has to be used for all sources", e.exception.__str__())

    def test_layer_only_at_one_source(self):
        """
        Illegal – `layer` syntax used but only for one source. If it's used, it has to be
        applied to all sources.
        """
        yaml_input = """
        definition:
          - merge_method: slerp
            sources:
              - model: A
                base_model: True
                layer: "model.layers.0.self_attn.k_proj.weight" 
              - model: B
          """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("If used, `layer` has to be used for all sources", e.exception.__str__())

    def test_both_range_and_layer_at_the_slice_level(self):
        """
        Illegal – `layer` syntax used but only for one source. If it's used, it has to be
        applied to all sources.
        """
        yaml_input = """
        definition:
          - merge_method: slerp
            layer: "asd"
            range: [0, 5]
            sources:
              - model: A
                base_model: True 
              - model: B
          """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("Cannot have both 'range' and 'layer' at the top level of the slice", e.exception.__str__())

    def test_slices_mixed_layer_range_syntax(self):
        """
        Illegal – `layer` syntax used but only for one source. If it's used, it has to be
        applied to all sources.
        """
        yaml_input = """
        definition:
          - merge_method: slerp
            sources:
              - model: A
                base_model: True
                range: [0, 1]
              - model: B
                layer: "Asd"
          """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("Slice sources have to be defined with either `layer` or `range`, not both",
                         e.exception.__str__())

    def test_layers_filter_used_with_layer_syntax(self):
        """
        Illegal – `layers` filter can only be used with `range` syntax, not with `layer` syntax
        """
        yaml_input = """
        definition:
          - sources:
              - model: A
                layer: "asd"
              - model: B
                layer: "asd"
            layers: ["xyz"]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("Layers filter can only be used with `range` definition, not with `layer`",
                         e.exception.__str__())

    def test_source_range_not_positive(self):
        """
        Illegal – `layers` filter can only be used with `range` syntax, not with `layer` syntax
        """
        yaml_input = """
        definition:
          - sources:
              - model: A
                range: [100, 0]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("Provided layers range is not positive", e.exception.__str__())

    def test_source_range_at_least_length_one(self):
        """
        Illegal – `layers` filter can only be used with `range` syntax, not with `layer` syntax
        """
        yaml_input = """
        definition:
          - sources:
              - model: A
                range: [0, 0]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("Provided layers range is not positive", e.exception.__str__())

    def test_source_ranges_the_same_length(self):
        """
        Illegal – `layers` filter can only be used with `range` syntax, not with `layer` syntax
        """
        yaml_input = """
        definition:
          - sources:
              - model: A
                range: [0, 1]
              - model: A
                range: [0, 999]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        validator = SliceValidator()

        with self.assertRaises(Exception) as e:
            validator.validate(yaml_loaded["definition"][0])

        self.assertEqual("All `range` must be of the same length", e.exception.__str__())
