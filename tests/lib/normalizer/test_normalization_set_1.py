import unittest

import yaml
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from unittest.mock import patch


class TestNormalizationRunner(unittest.TestCase):
    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def setUp(self, mock_load_architecture):
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
            ]
        }
        self.runner = NormalizationRunner("dummy_path.json")

    def test_same_range_at_sources_level(self):
        """
        Valid configuration – same length `range` in both sources
        """
        yaml_input = """
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
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.1.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        # Should work without raising exception
        processed = []
        try:
            processed = self.runner.normalize(yaml_loaded)
        except Exception as e:
            self.assertEqual(e, None)

        self.assertEqual(expected, processed)

    def test_different_range_at_sources_level(self):
        """
        Valid configuration – different `range` values in both sources, but still the same length in both.
        """
        yaml_input = """
        - merge_method: slerp
          sources:
            - model: A
              base_model: True
              range: [0, 1]
            - model: B
              range: [5, 6]
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.5.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 1,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.1.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.6.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        # Should work without raising exception
        processed = []
        try:
            processed = self.runner.normalize(yaml_loaded)
        except Exception as e:
            self.assertEqual(e, None)

        self.assertEqual(expected, processed)

    def test_layer_only_at_once_source(self):
        """
        Illegal – `layer` syntax used but only for one source. If it's used, it has to be
        applied to all sources.
        """
        yaml_input = """
        - merge_method: slerp
          sources:
            - model: A
              base_model: True
              layer: "model.layers.0.self_attn.k_proj.weight" 
            - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)

        self.assertEqual("'layer' attribute must be defined for all sources", e.__str__())

    def test_range_indicated_only_at_once_source(self):
        """
        Illegal  – `range` syntax applied to one source only. If applied, must be applied
        at top level or to all sources (and the length must be the same for all `range`s).
        """
        yaml_input = """
        - merge_method: slerp
          sources:
            - model: A
              base_model: True
              range: [0, 1] 
            - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception, msg="'range' attribute of all sources must have the same length"):
            self.runner.normalize(yaml_loaded)

    def test_no_slices_defined(self):
        """
        Illegal – no slices defined in the config
        """
        with self.assertRaises(Exception, msg="at least one slice configuration must be provided"):
            self.runner.normalize([])

    def test_no_base_model_defined(self):
        """
        No base model defined, neither at top level or in sources.
        In that scenario, take the first model as a base one.
        """
        yaml_input = """
            - merge_method: slerp
              sources:
                - model: A
                  layer: "model.layers.0.self_attn.k_proj.weight"
                - model: B
                  layer: "model.layers.0.self_attn.k_proj.weight"
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)

        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_all_models_not_base(self):
        """
        No base model defined, neither at top level or in sources.
        In that scenario, take the first model as a base one.
        """
        yaml_input = """
            - merge_method: slerp
              range: [0, 1]
              sources:
                - model: A
                  base_model: False
                - model: B
                  base_model: False
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)
        self.assertEqual("No valid source found to set as base_model", e.exception.__str__())

    def test_top_level_base_model(self):
        """
        Base model name defined in the slice configuration instead of sources. Valid behavior.
        Model B should be marked as base, second one is taken specifically to test that
        the "select-first-source-as-base-model" behavior is not applied in this case.
        """
        yaml_input = """
            - merge_method: slerp
              range: [0, 1]
              base_model: B
              sources:
                - model: A
                - model: B
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"layer": "model.layers.0.self_attn.k_proj.weight", "model": "A", },
                        {"base_model": True, "layer": "model.layers.0.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)

        processed = self.runner.normalize(yaml_loaded)
        self.assertEqual(expected, processed)

    def test_top_level_base_model_conflicting_with_sources(self):
        """
        Top-level base_model configuration conflicts with base model selected in `sources`
        """
        yaml_input = """
          - merge_method: slerp
            base_model: B
            sources:
              - model: A
                base_model: True
              - model: B
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        with self.assertRaises(Exception) as e:
            self.runner.normalize(yaml_loaded)
        self.assertEqual(
            "Conflicting base_model: model 'B' designated at slice level but model 'A' designed at source level",
            e.exception.__str__()
        )

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_layer_defined(self, mock_load_architecture):
        """
        Specific layer names defined at source level.
        The output should fill in the other layers (mlp in this case) in the same output block,
        indicated by `index` value (should be `0` in both output slices).
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
            - merge_method: slerp
              sources:
                - model: A
                  layer: "model.layers.12.self_attn.k_proj.weight"
                - model: B
                  layer: "model.layers.12.self_attn.k_proj.weight"
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.12.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.mlp", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        runner = NormalizationRunner("dummy_path.json")

        processed = runner.normalize(yaml_loaded)
        self.assertEqual(2, len(processed))
        self.assertEqual(expected, processed)

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_layers_defined_with_layers_filter_conflict(self, mock_load_architecture):
        """
        Specific layer names defined at source level.
        The output should fill in the other layers (mlp in this case) in the same output block,
        indicated by `index` value (should be `0` in both output slices).
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
            - merge_method: slerp
              sources:
                - model: A
                  layer: "model.layers.12.self_attn.k_proj.weight"
                - model: B
                  layer: "model.layers.12.self_attn.k_proj.weight"
              layers: ["mlp"]
        """

        yaml_loaded = yaml.safe_load(yaml_input)
        runner = NormalizationRunner("dummy_path.json")

        with self.assertRaises(Exception) as e:
            runner.normalize(yaml_loaded)
        self.assertEqual(
            "Layers specified in the sources do not contain the layers specified in the filter",
            e.exception.__str__()
        )

    @patch('flow_merge.lib.loaders.normalizer.load_architecture')
    def test_layer_defined_with_layers_filter(self, mock_load_architecture):
        """
        Specific layer names defined at source level AND layers filter applied, but in non-conflicting way.
        The output should contain two slices, one with the source-specified layer (self_attn)
        and second as a passthrough of mlp layer from base model.
        In that case the `layers` filter has no practical effect.
        """
        mock_load_architecture.return_value = {
            "weights": [
                {"name": "model.layers.{layer_index}.self_attn.k_proj.weight", "type": "attn"},
                {"name": "model.layers.{layer_index}.mlp.weight", "type": "mlp"},
            ]
        }
        yaml_input = """
            - merge_method: slerp
              sources:
                - model: A
                  layer: "model.layers.12.self_attn.k_proj.weight"
                - model: B
                  layer: "model.layers.12.self_attn.k_proj.weight"
              layers: ["self_attn"]
        """
        expected = [
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.self_attn.k_proj.weight", "model": "A", },
                        {"layer": "model.layers.12.self_attn.k_proj.weight", "model": "B", },
                    ],
                    "merge_method": "slerp"
                },
            },
            {
                "index": 0,
                "slice": {
                    "sources": [
                        {"base_model": True, "layer": "model.layers.12.mlp", "model": "A", },
                    ],
                    "merge_method": "passthrough"
                },
            },
        ]

        yaml_loaded = yaml.safe_load(yaml_input)
        runner = NormalizationRunner("dummy_path.json")

        processed = runner.normalize(yaml_loaded)
        self.assertEqual(2, len(processed))
        self.assertEqual(expected, processed)
