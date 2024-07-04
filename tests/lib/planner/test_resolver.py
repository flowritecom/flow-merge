import unittest
from unittest.mock import Mock
from flow_merge.lib.planners.resolver import extract_models_by_layers, ModelLayers

class TestResolver(unittest.TestCase):

    def test_valid_data(self):
        slices = [
            {'index': 0, 'slice': {'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.embed_tokens.weight', 'model': 'model_1'}]}},
            {'index': 7, 'slice': {'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_1'}, {'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_2'}]}},
            {'index': 28, 'slice': {'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.norm.weight', 'model': 'model_1'}, {'layer': 'model.norm.weight', 'model': 'model_2'}]}},
        ]

        logger = Mock()
        
        expected_output = ModelLayers(
            base_model='model_1',
            models={
                'model_2': ['model.layers.0.mlp.gate_proj.weight', 'model.norm.weight']
            }
        )

        result = extract_models_by_layers(slices, logger)
        self.assertEqual(result.base_model, expected_output.base_model)
        self.assertEqual(set(result.models.keys()), set(expected_output.models.keys()))
        for model in result.models:
            self.assertCountEqual(result.models[model], expected_output.models[model])

    def test_no_base_model(self):
        slices = [
            {'index': 0, 'slice': {'merge_method': 'passthrough', 'sources': [{'layer': 'model.embed_tokens.weight', 'model': 'model_1'}]}},
            {'index': 7, 'slice': {'merge_method': 'passthrough', 'sources': [{'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_1'}, {'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_2'}]}},
        ]

        logger = Mock()

        result = extract_models_by_layers(slices, logger)
        self.assertIsNone(result.base_model)
        self.assertEqual(set(result.models.keys()), set(["model_1", "model_2"]))
        self.assertCountEqual(result.models["model_1"], ['model.embed_tokens.weight', 'model.layers.0.mlp.gate_proj.weight'])
        self.assertCountEqual(result.models["model_2"], ['model.layers.0.mlp.gate_proj.weight'])

    def test_empty_slices(self):
        slices = []
        
        with self.assertRaises(TypeError):
            extract_models_by_layers(slices)
    
    def test_invalid_data_structure(self):
        slices = [
            {'index': 0, 'slice': {'merge_method': 'passthrough', 'sources': 'invalid_data'}}
        ]
        
        with self.assertRaises(TypeError):
            extract_models_by_layers(slices)
    
    def test_missing_keys(self):
        slices = [
            {'index': 0, 'slice': {'merge_method': 'passthrough', 'sources': [{}]}}
        ]
        
        with self.assertRaises(TypeError):
            extract_models_by_layers(slices)

if __name__ == '__main__':
    unittest.main()