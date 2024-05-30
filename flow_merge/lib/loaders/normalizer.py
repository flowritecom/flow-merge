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

    def ensure_base_model(self, slice: Dict[str, Any]) -> Dict[str, Any]:
        sources = slice.get("sources", [])
        if not any((src.get("base_model") == True) for src in sources):
            sources[0]["base_model"] = True
        slice["sources"] = sources
        return slice

    def create_slice_entries(
        self, sources: List[Dict[str, Any]], layer: str, merge_method: str
    ) -> Dict[str, Any]:
        return {
            "slice": {
                "merge_method": merge_method,
                "sources": [{**src, "layer": layer} for src in sources],
            }
        }

    def create_slices_from_range(
        self,
        sources: List[Dict[str, Any]],
        start: int,
        end: int,
        merge_method: str,
        layers_to_keep: List[str],
    ) -> List[Dict[str, Any]]:
        return [
            self.create_slice_entries(sources, lt.format(layer_index=i), merge_method)
            for i in range(start, end + 1)
            for lt, lta in self.layers.items()
            if layers_to_keep is None or any(lt in lta for lt in layers_to_keep)
        ]

    def expand_range(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.create_slices_from_range(
            config["sources"],
            config["range"][0],
            config["range"][1],
            config.get("merge_method"),
            config.get("layers", None),
        )

    def process_layer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.create_slice_entries(
            config["sources"], config["layer"], config.get("merge_method")
        )

    def apply_transformations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return reduce(lambda c, t: t(c), self.transformations, config)

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_data = []
        for config in raw_data:
            config = self.apply_transformations(config)
            if "range" in config:
                normalized_data.extend(self.expand_range(config))
            elif "layer" in config:
                normalized_data.append(self.process_layer(config))
            else:
                normalized_data.append(config)
        return normalized_data

def test_expand_range(
    config: Dict[str, Any], runner: NormalizationRunner
) -> List[Dict[str, Any]]:
    config = runner.apply_transformations(config)
    return runner.expand_range(config)


def display_slices(slices: List[Dict[str, Any]]):
    for entry in slices:
        print(json.dumps(entry, indent=2))
