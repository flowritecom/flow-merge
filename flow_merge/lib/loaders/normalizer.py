import json
from typing import Any, Dict, List, Optional
from functools import reduce
import re

from importlib import resources


def load_architecture(file_path: str) -> Dict[str, Any]:
    with resources.open_text("flow_merge.data.architectures", file_path) as file:
        return json.load(file)


class NormalizationRunner:
    models_layers: Dict[str, Dict[str, Any]] = {}
    models_layers_by_type: Dict[str, Dict[str, List[str]]] = {}

    def __init__(self):
        self.transformations = [self._ensure_base_model]

    def normalize(self, raw_data: Dict) -> List[Dict[str, Any]]:
        self._load_models_layers(raw_data)

        if "base_model" not in raw_data:
            raise ValueError("Base model is missing")

        slices = raw_data["definition"]
        normalized_data = []
        for i, s in enumerate(slices):
            s = self._apply_transformations(s)
            s["output_layer_id"] = i
            normalized_data.extend(self._process_slice(s))
        normalized_data = self._process_special_layers(normalized_data, raw_data["base_model"])
        normalized_data = self._move_embed_slice_to_top(normalized_data)
        normalized_data = self._reindex_slices_with_embed_slice(normalized_data)
        return normalized_data

    def _apply_transformations(self, slice: Dict[str, Any]) -> Dict[str, Any]:
        # ideally we want transformations being able to be passed in
        # from outside of the Normalizer class to customize behavior
        # these transformations are used to edit the slice not the
        # whole list of slices
        return reduce(lambda c, t: t(c), self.transformations, slice)

    def _ensure_base_model(self, slice: Dict[str, Any]) -> Dict[str, Any]:
        # we allow user to write slice without indicating base_model
        # or just base_model = False
        # here we make sure every slice has one base_model = True
        # where it is picked to be the first non-False, if undecided
        sources = slice.get("sources", [])

        # check if there's any source with base_model set to True
        if not any((src.get("base_model") is True) for src in sources):
            # attempt to find the first source that isn't explicitly marked as base_model = False
            # and set that to be the base_model = True
            for src in sources:
                if src.get("base_model") is not False:
                    src["base_model"] = True
                    slice["sources"] = sources
                    return slice
            raise ValueError("No valid source found to set as base_model")
        # if already a source with base_model == True, return the original slice
        return slice

    def _process_slice(self, slice: Dict[str, Any]) -> List[Dict[str, Any]]:
        # we process layer and range type slices, expanding range type
        slices = self._process_template_slices(slice)

        return slices

    def _process_special_layers(self, normalized_data, base_model: str) -> List[Dict[str, Any]]:
        # we don't want to hard code what the special layers are so we
        # say that the special layers are the ones that aren't templated
        # with an iterator
        # in most cases model.embed_tokens, model.norm and lm_head
        # we only look for 'embed', 'norm' and 'lm_head' to find them
        # if we don't find any of them then we just produce an empty list

        # the special layers clone the sources data from the adjacent
        # layer to them, in this case embed layer clones the first layer
        # (FIX: we don't impose which layer type it should look for cloning)
        # and the last ones use the last non-special layer

        # Removes all unnecessary attributes for special layer slices sources
        def get_plain_sources(sources: List[Dict]):
            return [
                {"model": source["model"], **({"base_model": True} if "base_model" in source else {})}
                for source in sources
            ]

        special_layer_names = [
            item
            for item in list(self.models_layers[base_model].keys())
            if "{" not in item and "}" not in item
        ]

        for special_layer_name in special_layer_names:
            if "embed" in special_layer_name:
                embed_slice = self._create_slice(
                    get_plain_sources(normalized_data[0]["slice"]["sources"]),
                    special_layer_name,
                    "interpolate",
                    0
                )
                normalized_data.append(embed_slice)

            if "norm" in special_layer_name:
                norm_slice = self._create_slice(
                    get_plain_sources(normalized_data[len(normalized_data) - 1]["slice"]["sources"]),
                    special_layer_name,
                    "interpolate",
                    self._get_last_output_slice_id(normalized_data)+1
                )
                normalized_data.append(norm_slice)

            if "lm_head" in special_layer_name:
                lm_head_slice = self._create_slice(
                    get_plain_sources(normalized_data[len(normalized_data) - 1]["slice"]["sources"]),
                    special_layer_name,
                    "interpolate",
                    self._get_last_output_slice_id(normalized_data)+1
                )
                normalized_data.append(lm_head_slice)
        return normalized_data

    def _get_last_output_slice_id(self, slices: List[Dict[str, Any]]) -> int:
        sorted_slices = sorted(slices, key=lambda s: -s["output_layer_id"])
        return sorted_slices[0]["output_layer_id"]

    def _process_template_slices(self, slice: Dict[str, Any]) -> List[Dict[str, Any]]:
        # should only be passed non-special-layer slices
        # ie. ones that take layer index to be templated and thus are expanded
        # to match the range indicated - or ones where a single 'layer' is
        # already specified manually (below)
        base_model = self._determine_base_model(slice["sources"])
        base_source = self._determine_base_source(slice["sources"])
        layer_name_templates = [
            item for item in list(self.models_layers[base_model]) if "{" in item and "}" in item
        ]

        if all("range" in src for src in slice["sources"]) and "layers" not in slice:
            start, end = slice["sources"][0]["range"]
            return [
                {
                    "output_layer_id": slice["output_layer_id"] + i,
                    "slice": {
                        "merge_method": slice["merge_method"],
                        "sources": [
                            {
                                **({k: v for k, v in src.items() if k != "range"}),
                                **({"layer": lnt.format(layer_index=src["range"][0] + i)})
                            }
                            for src in slice["sources"]
                        ],
                    }
                }
                for i in range(end - start + 1)
                for lnt in layer_name_templates
            ]
        elif all("range" in src for src in slice["sources"]) and "layers" in slice:
            # Layers filter applied

            # Validate if requested layers are at all available in the architecture of the base model
            for l in slice["layers"]:
                if l not in self.models_layers_by_type[base_model]:
                    raise Exception(f"Layer '{l}' does not exist in the model")

            start, end = slice["sources"][0]["range"]
            user_requested_layers = [
                layer
                for t, layers in self.models_layers_by_type[base_model].items() if t in slice["layers"]
                for layer in layers
            ]
            remaining_layers = list(set(self.models_layers[base_model]) - set(user_requested_layers))

            user_requested_slices = [
                {
                    "output_layer_id": slice["output_layer_id"] + i,
                    "slice": {
                        "merge_method": slice["merge_method"],
                        "sources": [
                            {
                                **({k: v for k, v in src.items() if k != "range"}),
                                **({"layer": lnt.format(layer_index=src["range"][0] + i)})
                            }
                            for src in slice["sources"]
                        ],
                    }
                }
                for i in range(end - start + 1)
                for lnt in user_requested_layers
            ]
            remaining_slices = [
                {
                    "output_layer_id": slice["output_layer_id"] + i,
                    "slice": {
                        "merge_method": "passthrough",
                        "sources": [{"model": base_model, "base_model": True,
                                     "layer": lnt.format(layer_index=base_source["range"][0] + i)}],
                    }
                }
                for i in range(end - start + 1)
                for lnt in remaining_layers
            ]

            return user_requested_slices + remaining_slices
        elif all("layer" in src for src in slice["sources"]):
            user_defined_layer_id = re.findall(r'\.(\d+)\.', base_source["layer"])
            if len(user_defined_layer_id) == 0:
                raise Exception("Layer defined for merging must be a hidden layer (pattern layer)")
            user_defined_layer = re.sub(r'\.\d+\.', ".{layer_index}.", base_source["layer"])
            remaining_layers = [l for l in self.models_layers[base_model] if l != user_defined_layer]

            user_defined_slice = [
                self._create_slice(slice["sources"], None, slice["merge_method"], slice["output_layer_id"])]
            remaining_slices = [
                self._create_slice([base_source], layer.format(layer_index=user_defined_layer_id[0]), "passthrough",
                                   slice["output_layer_id"])
                for layer in remaining_layers
            ]

            return user_defined_slice + remaining_slices

        raise Exception("Neither range or layers defined for merging")

    def _create_slice(
            self, sources: List[Dict[str, Any]], layer: Optional[str], merge_method: str, output_layer_id: int
    ) -> Dict[str, Any]:
        # creates a slice, sets merge_method and layer
        # while making sure to keep all other keys.
        # Determines if 'layer' is specified in any source,
        # if not, use the higher-level layer
        sources_with_layer = [
            {**src, **({"layer": layer} if layer or "layer" not in src else {})}
            for src in sources
        ]
        return {
            "output_layer_id": output_layer_id,
            "slice": {
                "merge_method": merge_method,
                "sources": sources_with_layer,
            }
        }

    def _determine_base_model(self, sources: List[Dict[str, Any]]) -> str | None:
        # determine which model name is the base_model for the given slice
        base_source = self._determine_base_source(sources)
        return base_source["model"] if base_source is not None else None

    def _determine_base_source(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Every slice must have base source"""
        for src in sources:
            if src.get("base_model", False):
                return src

    def _load_models_layers(self, raw_data: Dict[str, Any]):
        all_models = [raw_data["base_model"]] if "base_model" in raw_data else []
        all_models.extend([
            src["model"]
            for s in raw_data["definition"]
            for src in s["sources"]
        ])

        for m in all_models:
            arch = load_architecture(m)
            self.models_layers[m] = {
                weight["name"]: weight["type"] for weight in arch["weights"]
            }

            # Group weights in type groups
            self.models_layers_by_type[m] = {
                weight["type"]: [
                    w["name"] for w in arch["weights"] if w["type"] is weight["type"]
                ]
                for weight in arch["weights"]
            }

    def _move_embed_slice_to_top(self, normalized_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embed_index = self._embed_slice_index(normalized_data)
        if embed_index is None:
            return normalized_data

        normalized_data.insert(0, normalized_data.pop(embed_index))
        return normalized_data

    def _reindex_slices_with_embed_slice(self, normalized_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embed_index = self._embed_slice_index(normalized_data)
        if embed_index is None:
            return normalized_data

        for i, slice in enumerate(normalized_data):
            if i is not embed_index:
                slice["output_layer_id"] += 1

        return normalized_data

    def _embed_slice_index(self, normalized_data: List[Dict[str, Any]]) -> bool | int:
        return next(
            (
                i
                for i, slice_entry in enumerate(normalized_data)
                if any("embed" in src.get("layer", "") for src in slice_entry["slice"]["sources"])
            ),
            None,
        )


