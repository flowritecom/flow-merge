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
        self.transformations = [self._ensure_base_model]

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_data = []
        for slice in raw_data:
            slice = self._apply_transformations(slice)
            normalized_data.extend(self._process_slice(slice, normalized_data))
        normalized_data += self._process_special_layers(normalized_data)
        normalized_data = self._move_embed_slice_to_top(normalized_data)
        normalized_data = self._add_indices_to_slices(normalized_data)
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
        if not any((src.get("base_model") == True) for src in sources):
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

    def _create_slice(
        self, sources: List[Dict[str, Any]], layer: str, merge_method: str
    ) -> Dict[str, Any]:
        # creates a slice, sets merge_method and layer
        # while making sure to keep all other keys
        return {
            "slice": {
                "merge_method": merge_method,
                "sources": [{**src, "layer": layer} for src in sources],
            }
        }

    def _process_template_slices(self, slice: Dict[str, Any]) -> List[Dict[str, Any]]:
        # should only be passed non-special-layer slices
        # ie. ones that take layer index to be templated and thus are expanded
        # to match the range indicated - or ones where a single 'layer' is
        # already specified manually (below)
        layer_name_templates = [
            item for item in list(self.layers.keys()) if "{" in item and "}" in item
        ]
        if "range" in slice:
            start, end = slice["range"]
            return [
                self._create_slice(
                    slice["sources"], lnt.format(layer_index=i), slice["merge_method"]
                )
                for i in range(start, end + 1)
                for lnt in layer_name_templates
            ]
        elif "layer" in slice:
            return [
                self._create_slice(
                    slice["sources"], slice["layer"], slice["merge_method"]
                )
            ]
        return []

    def _update_sources_with_base_model(
        self,
        sources: List[Dict[str, Any]],
        base_model: str,
        layers_to_process: List[str],
    ) -> List[Dict[str, Any]]:
        # keep only the sources that are indicated with base_model = True,
        # those are the ones we are interested in mutating to be passthrough
        # from the base_model, if 'layers' filter is in place
        # - note. before this step we have created everything ready to be filtered
        new_sources = []
        for src in sources:
            if any(layer_type in src["layer"] for layer_type in layers_to_process):
                src["model"] = base_model
                if src.get("base_model", False):
                    new_sources.append(src)
            else:
                new_sources.append(src)
        return new_sources

    def _determine_base_model(self, sources: List[Dict[str, Any]]) -> str:
        # determine which model name is the base_model for the given slice
        for src in sources:
            if src.get("base_model", False):
                return src["model"]
        return None

    def _edit_unfiltered_slices(
        self, slices: List[Dict[str, Any]], slice: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # if 'layers' filter is in place, edit the layer slices that aren't in the filter
        # to be passthrough from base_model
        # while making sure not to do this for the special_layer_names (embed_tokens, norm, lm_head)
        # also take those layer types dynamically from the architecture
        layer_types = list(self.layers.values())
        special_layer_names = [
            item
            for item in list(self.layers.keys())
            if "{" not in item and "}" not in item
        ]
        layers_to_keep = slice.get("layers", [])
        layers_to_process = [
            lt for lt in layer_types if lt not in layers_to_keep + special_layer_names
        ]

        for slice_entry in slices:
            base_model = self._determine_base_model(slice_entry["slice"]["sources"])
            slice_entry["slice"]["sources"] = self._update_sources_with_base_model(
                slice_entry["slice"]["sources"], base_model, layers_to_process
            )
            slice_entry["slice"]["merge_method"] = "passthrough"

        return slices

    def _process_slice(
        self, slice: Dict[str, Any], normalized_data
    ) -> List[Dict[str, Any]]:
        # we process layer and range type slices, expanding range type
        slices = self._process_template_slices(slice)
        # we take filter condition 'layers' into consideration
        # and change the corresponding layers to passthrough and drop sources
        # that are not the indicated base_model of the slice
        slices = self._edit_unfiltered_slices(slices, slice)
        return slices

    def _process_special_layers(self, normalized_data) -> List[Dict[str, Any]]:
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
        # (FIX: we should make sure which sources are copied directly when
        # cloning and which ones aren't, for example range or layer
        # shouldn't logically be cloned)
        special_layer_names = [
            item
            for item in list(self.layers.keys())
            if "{" not in item and "}" not in item
        ]
        for special_layer_name in special_layer_names:
            if "embed" in special_layer_name:
                embed_slice = self._create_slice(
                    normalized_data[0]["slice"]["sources"],
                    special_layer_name,
                    normalized_data[0]["slice"]["merge_method"],
                )
            if "norm" in special_layer_name:
                norm_slice = self._create_slice(
                    normalized_data[len(normalized_data) - 1]["slice"]["sources"],
                    special_layer_name,
                    normalized_data[len(normalized_data) - 1]["slice"]["merge_method"],
                )
            if "lm_head" in special_layer_name:
                lm_head_slice = self._create_slice(
                    normalized_data[len(normalized_data) - 1]["slice"]["sources"],
                    special_layer_name,
                    normalized_data[len(normalized_data) - 1]["slice"]["merge_method"],
                )
        slices = [embed_slice, norm_slice, lm_head_slice]
        return slices

    def _move_embed_slice_to_top(
        self, normalized_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # we added the embed_tokens layer when we didn't yet have and index
        # so we move it to the beginning, the other special layers will remain at the end
        # with ordering norm and lm_head
        embed_index = next(
            i
            for i, slice_entry in enumerate(normalized_data)
            if any("embed" in src["layer"] for src in slice_entry["slice"]["sources"])
        )
        normalized_data.insert(0, normalized_data.pop(embed_index))
        return normalized_data

    def _add_indices_to_slices(
        self, normalized_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # we create the index as the last thing
        return [
            {**slice_entry, "index": idx}
            for idx, slice_entry in enumerate(normalized_data)
        ]


def display_slices(slices: List[Dict[str, Any]]):
    for entry in slices:
        print(json.dumps(entry, indent=2))


#################
## TODO: option to include range and layer in the sources directly \
##       but have top level as option too to apply for them

## TODO: allow shorter version of the template language if filter is present
