import json
from typing import Any, Dict, List
from functools import reduce
import pkg_resources


def load_architecture(file_path: str) -> Dict[str, Any]:
    resource_package = __name__
    resource_path = f"../../data/architectures/{file_path}"
    file_content = pkg_resources.resource_string(resource_package, resource_path)
    return json.loads(file_content)


class NormalizationRunner:
    def __init__(self, architecture_path: str):
        self.architecture = load_architecture(architecture_path)
        self.layers = {
            weight["name"]: weight["type"] for weight in self.architecture["weights"]
        }
        self.transformations = [self.ensure_base_model]

    def ensure_base_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        sources = config.get("sources", [])
        if not any((src.get("base_model") == True) for src in sources):
            sources[0]["base_model"] = True
        config["sources"] = sources
        return config

    def create_slice(
        self, sources: List[Dict[str, Any]], layer: str, merge_method: str
    ) -> Dict[str, Any]:
        return {
            "slice": {
                "merge_method": merge_method,
                "sources": [{**src, "layer": layer} for src in sources],
            }
        }

    def process_slice(self, slice: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources = slice["sources"]
        merge_method = slice.get("merge_method")
        layers_to_keep = slice.get("layers")
        layers = list(self.layers.keys())
        layer_types = list(self.layers.values())

        if "range" in slice:
            start, end = slice["range"]
            slices = [
                self.create_slice(sources, lt.format(layer_index=i), merge_method)
                for i in range(start, end + 1)
                for lt in layers
            ]
        elif "layer" in slice:
            slices = [self.create_slice(sources, slice["layer"], merge_method)]

        ## TODO: bring the missing layers from the base_model if we have a
        ## layer filter
        base_model = None
        for search_slice in slices:
            for src in search_slice["slice"]["sources"]:
                if src.get("base_model", False):
                    base_model = src.get("model")

        new_slices = []
        for write_slice in slices:
            for layer_type in layer_types:
                if (
                    layer_type in write_slice["slice"]["sources"][0]["layer"]
                    and layer_type in layers_to_keep
                ):
                    new_slices.append(write_slice)
                    continue
                else:
                    for layer_template_name in layers:
                        new_slices.append(
                            {
                                "slice": {
                                    "merge_method": "passthrough",
                                    "sources": [
                                        {
                                            "model": base_model,
                                            "base_model": True,
                                            "layer": layer_template_name,
                                        }
                                    ],
                                }
                            }
                        )

        return new_slices

    def apply_transformations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return reduce(lambda c, t: t(c), self.transformations, config)

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_data = []
        for config in raw_data:
            config = self.apply_transformations(config)
            normalized_data.extend(self.process_slice(config))
        return normalized_data


def test_expand_range(
    config: Dict[str, Any], runner: NormalizationRunner
) -> List[Dict[str, Any]]:
    config = runner.apply_transformations(config)
    return runner.handle_config(config)


def display_slices(slices: List[Dict[str, Any]]):
    for entry in slices:
        print(json.dumps(entry, indent=2))


#################
## TODO: option to include range and layer in the sources directly
## TODO: but have top level as option too to apply for them

## TODO: allow shorter version of the template language if filter is present
