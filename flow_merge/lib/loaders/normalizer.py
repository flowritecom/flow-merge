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

    def create_slice(
        self, sources: List[Dict[str, Any]], layer: str, merge_method: str
    ) -> Dict[str, Any]:
        return {
            "slice": {
                "merge_method": merge_method,
                "sources": [{**src, "layer": layer} for src in sources],
            }
        }

    def process_slice(
        self, slice: Dict[str, Any], normalized_data
    ) -> List[Dict[str, Any]]:
        special_layer_names = [
            item
            for item in list(self.layers.keys())
            if "{" not in item and "}" not in item
        ]
        layer_name_templates = [
            item for item in list(self.layers.keys()) if "{" in item and "}" in item
        ]
        layer_types = list(self.layers.values())

        if "range" in slice:
            start, end = slice["range"]
            slices = [
                self.create_slice(
                    slice["sources"], lnt.format(layer_index=i), slice["merge_method"]
                )
                for i in range(start, end + 1)
                for lnt in layer_name_templates
            ]
        elif "layer" in slice:
            slices = [
                self.create_slice(
                    slice["sources"], slice["layer"], slice["merge_method"]
                )
            ]
        base_model = None
        layers_to_keep = slice.get("layers", None)
        layers_to_process = list(
            set(
                [
                    lt
                    for lt in layer_types
                    if lt not in layers_to_keep + special_layer_names
                ]
            )
        )
        for i, slice_entry in enumerate(slices):
            new_sources = []
            for src in slice_entry["slice"]["sources"]:
                if src.get("base_model", False):
                    base_model = src["model"]
                if any(layer_type in src["layer"] for layer_type in layers_to_process):
                    src["model"] = base_model
                    slice_entry["slice"]["merge_method"] = "passthrough"
                    if src.get("base_model", False):
                        new_sources.append(src)
                else:
                    new_sources.append(src)

            slice_entry["slice"]["sources"] = new_sources
            slices[i] = slice_entry
        return slices

    def process_special_layers(self, normalized_data) -> List[Dict[str, Any]]:
        special_layer_names = [
            item
            for item in list(self.layers.keys())
            if "{" not in item and "}" not in item
        ]
        for special_layer_name in special_layer_names:
            if "embed" in special_layer_name:
                embed_slice = self.create_slice(
                        normalized_data[0]["slice"]["sources"],
                        special_layer_name,
                        normalized_data[0]["slice"]["merge_method"],
                    )
            if "norm" in special_layer_name:
                norm_slice = self.create_slice(
                        normalized_data[len(normalized_data)-1]["slice"]["sources"],
                        special_layer_name,
                        normalized_data[len(normalized_data)-1]["slice"]["merge_method"],
                    )
            if "lm_head" in special_layer_name:
                lm_head_slice = self.create_slice(
                    normalized_data[len(normalized_data)-1]["slice"]["sources"],
                    special_layer_name,
                    normalized_data[len(normalized_data)-1]["slice"]["merge_method"],
                    )
        slices = [embed_slice, norm_slice, lm_head_slice]
        return slices

    def apply_transformations(self, slice: Dict[str, Any]) -> Dict[str, Any]:
        return reduce(lambda c, t: t(c), self.transformations, slice)

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_data = []
        for slice in raw_data:
            slice = self.apply_transformations(slice)
            normalized_data.extend(self.process_slice(slice, normalized_data))
        normalized_data = normalized_data + self.process_special_layers(normalized_data)

        # Move slice containing "embed" in any source's layer to the top
        normalized_data.insert(0, normalized_data.pop(next(i for i, slice_entry in enumerate(normalized_data) if any("embed" in src["layer"] for src in slice_entry["slice"]["sources"]))))

        # Add index to each slice
        normalized_data = [{**slice_entry, "index": idx} for idx, slice_entry in enumerate(normalized_data)]

        return normalized_data


def display_slices(slices: List[Dict[str, Any]]):
    for entry in slices:
        print(json.dumps(entry, indent=2))


#################
## TODO: option to include range and layer in the sources directly
## TODO: but have top level as option too to apply for them

## TODO: allow shorter version of the template language if filter is present
