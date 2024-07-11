import json
from typing import Any, Dict, List, Optional
from functools import reduce
import re

from importlib import resources


def load_architecture(file_path: str) -> Dict[str, Any]:
    with resources.open_text("flow_merge.data.architectures", file_path) as file:
        return json.load(file)


class Source:
    layer: Optional[str]
    range: Optional[List[int]]
    model: str
    is_base: Optional[bool]

    def __init__(self, **kwargs):
        self.layer = kwargs["layer"] if "layer" in kwargs else None
        self.range = kwargs["range"] if "range" in kwargs else None
        self.model = kwargs["model"] if "model" in kwargs else None
        self.is_base = kwargs["is_base"] if "is_base" in kwargs else None

    def update(self, attr: str, value: Any):
        self.__setattr__(attr, value)
        return self


class Slice:
    output_layer_id: int
    layers: Optional[List[str]] = None
    sources: List[Source] = None
    merge_method: str

    def __init__(self, **kwargs):
        self.output_layer_id = kwargs["output_layer_id"] if "output_layer_id" in kwargs else None
        self.layers = kwargs["layers"] if "layers" in kwargs else None
        self.sources = [
            Source(**source) if isinstance(source, dict) else Source(**source.__dict__)
            for source in kwargs["sources"]
        ]
        self.merge_method = kwargs["merge_method"] if "merge_method" in kwargs else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            **{"sources": [s.__dict__ for s in self.sources]}
        }


class NormalizationRunner:
    models_layers: Dict[str, Dict[str, Any]] = {}
    models_layers_by_type: Dict[str, Dict[str, List[str]]] = {}

    def __init__(self):
        self.transformations = [self._ensure_base_model]

    def normalize(self, raw_data: Dict) -> List[Dict[str, Any]]:
        self._load_models_layers(raw_data)

        if "base_model" not in raw_data:
            raise ValueError("Base model is missing")

        slices = [Slice(**s) for s in raw_data["definition"]]
        normalized_slices = []
        for i, s in enumerate(slices):
            s = self._apply_transformations(s)
            s.output_layer_id = i
            normalized_slices.extend(self._process_slice(s))
        normalized_slices = self._process_special_layers(normalized_slices, raw_data["base_model"])
        normalized_slices = self._move_embed_slice_to_top(normalized_slices)
        normalized_slices = self._reindex_slices_with_embed_slice(normalized_slices)

        for s in normalized_slices:
            s.__delattr__("layers")
            for src in s.sources:
                src.__delattr__("range")
                if src.is_base is None:
                    src.__delattr__("is_base")
        return [s.to_dict() for s in normalized_slices]

    def _apply_transformations(self, s: Slice) -> Slice:
        # ideally we want transformations being able to be passed in
        # from outside of the Normalizer class to customize behavior
        # these transformations are used to edit the slice not the
        # whole list of slices
        return reduce(lambda c, t: t(c), self.transformations, s)

    def _ensure_base_model(self, slice: Slice) -> Slice:
        # we allow user to write slice without indicating base_model
        # or just base_model = False
        # here we make sure every slice has one base_model = True
        # where it is picked to be the first non-False, if undecided
        sources = slice.sources or []

        # check if there's any source with base_model set to True
        if not any((src.is_base is True) for src in sources):
            # attempt to find the first source that isn't explicitly marked as base_model = False
            # and set that to be the base_model = True
            for idx, src in enumerate(sources):
                if src.is_base is not False:
                    slice.sources[idx].is_base = True
                    return slice
            raise ValueError("No valid source found to set as base_model")
        # if already a source with base_model == True, return the original slice
        return slice

    def _process_slice(self, s: Slice) -> List[Slice]:
        # we process layer and range type slices, expanding range type
        slices = self._process_template_slices(s)

        return slices

    def _process_special_layers(self, normalized_data: List[Slice], base_model: str) -> List[Slice]:
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
        def get_plain_sources(sources: List[Source]) -> List[Source]:
            return [
                Source(model=source.model, is_base=source.is_base)
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
                    get_plain_sources(normalized_data[0].sources),
                    special_layer_name,
                    "interpolate",
                    0
                )
                normalized_data.append(embed_slice)

            if "norm" in special_layer_name:
                norm_slice = self._create_slice(
                    get_plain_sources(normalized_data[len(normalized_data) - 1].sources),
                    special_layer_name,
                    "interpolate",
                    self._get_last_output_slice_id(normalized_data) + 1
                )
                normalized_data.append(norm_slice)

            if "lm_head" in special_layer_name:
                lm_head_slice = self._create_slice(
                    get_plain_sources(normalized_data[len(normalized_data) - 1].sources),
                    special_layer_name,
                    "interpolate",
                    self._get_last_output_slice_id(normalized_data) + 1
                )
                normalized_data.append(lm_head_slice)
        return normalized_data

    def _get_last_output_slice_id(self, slices: List[Slice]) -> int:
        sorted_slices = sorted(slices, key=lambda s: s.output_layer_id, reverse=True)
        return sorted_slices[0].output_layer_id

    def _process_template_slices(self, slice: Slice) -> List[Slice]:
        # should only be passed non-special-layer slices
        # ie. ones that take layer index to be templated and thus are expanded
        # to match the range indicated - or ones where a single 'layer' is
        # already specified manually (below)
        base_model = self._determine_base_model(slice.sources)
        base_source = self._determine_base_source(slice.sources)
        layer_name_templates = [
            item for item in list(self.models_layers[base_model]) if "{" in item and "}" in item
        ]

        def get_slices_for_all_layers(start, end, _slice: Slice, layers):
            return [
                Slice(
                    output_layer_id=_slice.output_layer_id + i,
                    merge_method=_slice.merge_method,
                    sources=[
                        Source(**{**src.__dict__, **{"layer": lnt.format(layer_index=src.range[0] + i), "range": None}})
                        # src.update("layer", lnt.format(layer_index=src.range[0] + i)).update("range", None)
                        for src in _slice.sources
                    ],
                )
                for i in range(end - start + 1)
                for lnt in layers
            ]

        if all(src.range is not None for src in slice.sources) and slice.layers is None:
            start, end = slice.sources[0].range
            return get_slices_for_all_layers(start, end, slice, layer_name_templates)

        elif all(src.range is not None for src in slice.sources) and slice.layers is not None:
            # Layers filter applied
            # We create slices with the layers defined by user in the layers filter
            # and then fill in the rest of the layers based on the model architecture definition.

            # Validate if requested layers are at all available in the architecture of the base model
            self._validate_layers_filter_values(base_model, slice)

            start, end = slice.sources[0].range
            user_requested_layers = [
                layer
                for requested_layer_type in slice.layers
                for layer in self.models_layers_by_type[base_model][requested_layer_type]
            ]
            remaining_layers = list(set(self.models_layers[base_model]) - set(user_requested_layers))

            user_requested_slices = get_slices_for_all_layers(start, end, slice, user_requested_layers)
            remaining_slices = [
                Slice(
                    output_layer_id=slice.output_layer_id + i,
                    merge_method="passthrough",
                    sources=[
                        Source(model=base_model, is_base=True,
                               layer=lnt.format(layer_index=base_source.range[0] + i))
                    ],
                )
                for i in range(end - start + 1)
                for lnt in remaining_layers
            ]

            return user_requested_slices + remaining_slices
        elif all(src.layer is not None for src in slice.sources):
            user_defined_layer_id = re.findall(r'\.(\d+)\.', base_source.layer)
            if len(user_defined_layer_id) == 0:
                raise Exception("Layer defined for merging must be a hidden layer (pattern layer)")
            user_defined_layer = re.sub(r'\.\d+\.', ".{layer_index}.", base_source.layer)
            remaining_layers = [l for l in self.models_layers[base_model] if l != user_defined_layer]

            user_defined_slice = [
                self._create_slice(slice.sources, None, slice.merge_method, slice.output_layer_id)]
            remaining_slices = [
                self._create_slice(
                    [base_source],
                    layer.format(layer_index=user_defined_layer_id[0]),
                    "passthrough",
                    slice.output_layer_id)
                for layer in remaining_layers
            ]

            return user_defined_slice + remaining_slices

        raise Exception("Neither range or layers defined for merging")

    def _validate_layers_filter_values(self, base_model: str, slice: Slice):
        for l in slice.layers:
            if l not in self.models_layers_by_type[base_model]:
                raise Exception(f"Layer '{l}' does not exist in the model")

    def _create_slice(
            self, sources: List[Source], layer: Optional[str], merge_method: str, output_layer_id: int
    ) -> Slice:
        # creates a slice, sets merge_method and layer
        # while making sure to keep all other keys.
        # Determines if 'layer' is specified in any source,
        # if not, use the higher-level layer
        sources_with_layer = [
            Source(**{**src.__dict__, **{"layer": layer or src.layer}})
            for src in sources
        ]
        return Slice(
            output_layer_id=output_layer_id,
            merge_method=merge_method,
            sources=sources_with_layer,
        )

    def _determine_base_model(self, sources: List[Source]) -> str | None:
        # determine which model name is the base_model for the given slice
        base_source = self._determine_base_source(sources)
        return base_source.model if base_source is not None else None

    def _determine_base_source(self, sources: List[Source]) -> Source:
        """Every slice must have base source"""
        for src in sources:
            if src.is_base is True:
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

    def _move_embed_slice_to_top(self, normalized_data: List[Slice]) -> List[Slice]:
        embed_index = self._embed_slice_index(normalized_data)
        if embed_index is None:
            return normalized_data

        normalized_data.insert(0, normalized_data.pop(embed_index))
        return normalized_data

    def _reindex_slices_with_embed_slice(self, normalized_data: List[Slice]) -> List[Slice]:
        embed_index = self._embed_slice_index(normalized_data)
        if embed_index is None:
            return normalized_data

        for i, slice in enumerate(normalized_data):
            if i is not embed_index:
                slice.output_layer_id += 1

        return normalized_data

    def _embed_slice_index(self, normalized_data: List[Slice]) -> bool | int:
        return next(
            (
                i
                for i, slice_entry in enumerate(normalized_data)
                if any("embed" in src.layer for src in slice_entry.sources)
            ),
            None,
        )


def _expand_range_slice() -> List[Dict[str, Any]]:
    pass


def _expand_range_slice_with_layers_filter() -> List[Dict[str, Any]]:
    pass


def _expand_layer_slice() -> List[Dict[str, Any]]:
    pass
