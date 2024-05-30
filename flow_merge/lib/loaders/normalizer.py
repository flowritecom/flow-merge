import json
from typing import Any, Dict, List, Callable
from functools import reduce
import pkg_resources

# Read the architecture from a JSON file
def load_architecture(file_path: str) -> Dict[str, Any]:
    resource_package = __name__  # Name of the current module
    resource_path = f'../../data/architectures/{file_path}'
    file_content = pkg_resources.resource_string(resource_package, resource_path)
    return json.loads(file_content)

class NormalizationRunner:
    def __init__(self, architecture_path: str):
        # self.raw_data = raw_data
        # self.env = env
        # self.logger = logger
        self.architecture = load_architecture(architecture_path)
        self.layers = {weight['name']: weight['type'] for weight in self.architecture['weights']}
        self.transformations = [self.ensure_base_model]

    def ensure_base_model(self, slice_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure that the base_model is indicated in the sources. If not, choose the first one.
        """
        sources = slice_config.get("sources", [])
        base_model_present = any("base_model" in src for src in sources)

        if not base_model_present and sources:
            sources[0]["base_model"] = True

        return slice_config


    def create_slice_entries(self, model: str, i: int, merge_method: Any, layers_to_keep: List[str]) -> List[Dict[str, Any]]:
        """
        Create complete slice entries for a given model and index `i` using the architecture.
        Optionally filter layers based on the layers_to_keep list.
        """
        return [
            {
                "slice": {
                    "merge_method": merge_method,
                    "sources": [{"model": model, "layer": layer_template.format(model=model, layer_index=i)}]
                }
            }
            for layer_template, layer_type_arch in self.layers.items()
            if layers_to_keep is None or any(layer_type in layer_type_arch for layer_type in layers_to_keep)
        ]

    def expand_range(self, slice_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand the range notation into full notation using list comprehension.
        """
        range_start, range_end = slice_config['range']
        merge_method = slice_config.get("merge_method")
        layers_to_keep = slice_config.get("layers")
        return [
            entry
            for i in range(range_start, range_end + 1)
            for src in slice_config['sources']
            for entry in self.create_slice_entries(src['model'], i, merge_method, layers_to_keep)
        ]

    def apply_transformations(self, slice_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a series of transformations to the slice_config.
        """
        return reduce(lambda config, transform: transform(config), self.transformations, slice_config)

    def normalize(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize the entire raw_data dictionary, applying transformations and expanding any range notations.
        """
        normalized_data = []
        for _, value in raw_data.items():
            if 'range' in value:
                normalized_data.extend(self.expand_range(value))
            else:
                normalized_data.append(value)
        return normalized_data


def test_expand_range(slice_config: Dict[str, Any], runner: NormalizationRunner) -> List[Dict[str, Any]]:
    """
    Test the expand_range function with a given slice configuration.
    """
    slice_config = runner.apply_transformations(slice_config)
    return runner.expand_range(slice_config)


def display_slices(slices: List[Dict[str, Any]]):
    """
    Display the slices in a formatted manner.
    """
    for slice_entry in slices:
        print(json.dumps(slice_entry, indent=2))
