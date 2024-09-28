from pydantic import BaseModel

from flow_merge.lib.merge_methods import MergeMethodIdentifier
from flow_merge.lib.model.architecture import ModelArchitectureProvider, ModelWeight, ModelWeightLayerType
from typing import Any, Dict, List, Optional
from functools import reduce

from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import get_logger

logger = get_logger(__name__)


class NormalizedSource(BaseModel):
    weight: Optional[float] = None
    model: Optional[str]
    layer: Optional[str]
    is_base: Optional[bool] = False


class MergeMethod(BaseModel):
    name: MergeMethodIdentifier
    params: Optional[Dict[str, Any]] = None


class NormalizedSlice(BaseModel):
    merge_method: MergeMethod
    sources: List[NormalizedSource]
    block_id: int
    output_layer_name: str
    layer_type: str


class _Source:
    layer: Optional[str]
    range: Optional[List[int]]
    model: str
    is_base: Optional[bool]
    weight: float

    def __init__(self, **kwargs):
        self.layer = kwargs["layer"] if "layer" in kwargs else None
        self.range = kwargs["range"] if "range" in kwargs else None
        self.model = kwargs["model"] if "model" in kwargs else None
        self.is_base = kwargs["is_base"] if "is_base" in kwargs else None
        self.weight = kwargs["weight"] if "weight" in kwargs else 1.0

    def update(self, attr: str, value: Any):
        self.__setattr__(attr, value)
        return self

    def todict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


class _MergeMethod:
    name: str
    params: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            raise Exception("Missing 'name' parameter for merge method")

        self.name = kwargs["name"]
        self.params = kwargs["params"] if "params" in kwargs else None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


class _Slice:
    block_id: int
    output_layer_name: str
    layers: Optional[List[str]] = None
    sources: List[_Source] = None
    merge_method: _MergeMethod
    layer_type: str

    def __init__(self, **kwargs):
        self.output_layer_name = kwargs["output_layer_name"] if "output_layer_name" in kwargs else None
        self.block_id = kwargs["block_id"] if "block_id" in kwargs else None
        self.layers = kwargs["layers"] if "layers" in kwargs else None
        self.sources = [
            _Source(**source) if isinstance(source, dict) else _Source(**source.__dict__)
            for source in kwargs["sources"]
        ]
        self.merge_method = _MergeMethod(**kwargs["merge_method"]) if isinstance(kwargs["merge_method"], dict) else \
            kwargs["merge_method"]
        self.layer_type = kwargs["layer_type"] if "layer_type" in kwargs else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            **{"sources": [s.todict() for s in self.sources]},
            **{"merge_method": self.merge_method.to_dict()},
        }


class NormalizationRunner:
    model_arch_provider: ModelArchitectureProvider = None
    models_layers: Dict[str, Dict[str, ModelWeight]] = {}
    models_layers_by_type: Dict[str, Dict[str, List[ModelWeight]]] = {}
    config: ApplicationConfig

    def __init__(self, model_arch_provider: ModelArchitectureProvider):
        self.transformations = [self._ensure_base_model]
        self.model_arch_provider = model_arch_provider

    def normalize(self, raw_data: Dict) -> List[Dict[str, Any]]:
        self._load_models_layers(raw_data)

        if "base_model" not in raw_data:
            raise ValueError("Base model is missing")

        slices = [_Slice(**s) for s in raw_data["definition"]]
        normalized_slices = []

        logger.info("Normalizing merge config slices")
        last_block_id = 0
        for i, s in enumerate(slices):
            s = self._apply_transformations(s)
            s.block_id = last_block_id
            normalized_slices.extend(self._process_template_slices(s))
            last_block_id = max(normalized_slices, key=lambda x: x.block_id).block_id + 1

        normalized_slices = self._process_special_layers(normalized_slices, raw_data["base_model"])
        normalized_slices = self._process_post_norm_merge_method(normalized_slices)

        for s in normalized_slices:
            s.__delattr__("layers")
            for src in s.sources:
                src.__delattr__("range")
                if src.is_base is None:
                    src.__delattr__("is_base")

        logger.info(f"Normalization complete, total slices {len(normalized_slices)}")

        return [s.to_dict() for s in normalized_slices]

    def _apply_transformations(self, s: _Slice) -> _Slice:
        # ideally we want transformations being able to be passed in
        # from outside of the Normalizer class to customize behavior
        # these transformations are used to edit the slice not the
        # whole list of slices
        return reduce(lambda c, t: t(c), self.transformations, s)

    def _ensure_base_model(self, _slice: _Slice) -> _Slice:
        # we allow user to write slice without indicating base_model
        # or just base_model = False
        # here we make sure every slice has one base_model = True
        # where it is picked to be the first non-False, if undecided
        sources = _slice.sources or []

        # check if there's any source with base_model set to True
        if not any((src.is_base is True) for src in sources):
            # attempt to find the first source that isn't explicitly marked as base_model = False
            # and set that to be the base_model = True
            for idx, src in enumerate(sources):
                if src.is_base is not False:
                    _slice.sources[idx].is_base = True
                    return _slice
            raise ValueError("No valid source found to set as base_model")
        # if already a source with base_model == True, return the original slice
        return _slice

    def _process_post_norm_merge_method(self, normalized_slices: List[_Slice]) -> List[_Slice]:
        # We treat the model.norm.weight layer as a special layer and add it with the interpolate method
        # Then after processing and correct order of the slices we update the method for this slice
        # This is to preserve the correct merge method based on the config
        # We always append the norm layer to the end so the previous merge method that is not interpolate
        # should hold true
        for i, _slice in enumerate(normalized_slices):
            if _slice.output_layer_name == "model.norm.weight":

                max_block_id = max(s.block_id for s in normalized_slices if s.block_id is not None)
                last_non_interpolate_block = [s for s in normalized_slices if
                                              s.block_id == max_block_id and s.merge_method != MergeMethodIdentifier.INTERPOLATE].pop()
                if last_non_interpolate_block is not None:
                    _slice.merge_method = last_non_interpolate_block.merge_method
                    break
        return normalized_slices

    def _process_special_layers(self, normalized_data: List[_Slice], base_model: str) -> List[_Slice]:
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
        def get_plain_sources(sources: List[_Source]) -> List[_Source]:
            return [
                _Source(model=source.model, is_base=source.is_base, weight=source.weight)
                for source in sources
            ]

        def get_last_block_sources(normalized_slices: List[_Slice]) -> List[_Source]:
            last_block_id = max(s.block_id for s in normalized_slices if s.block_id is not None)
            sources = [s.sources for s in normalized_slices if s.block_id == last_block_id]
            return [s for sublist in sources for s in sublist]  # flatten the sources list

        def unique_sources(sources: List[_Source]) -> List[_Source]:
            return list({s.model: s for s in sources}.values())

        special_layers = [
            layer
            for name, layer in self.models_layers[base_model].items()
            if layer.layer_type != "decoder"
        ]

        for special_layer in special_layers:
            if special_layer.layer_type == "embedding":
                embed_slice = self._create_slice(
                    get_plain_sources(normalized_data[0].sources),
                    special_layer.name,
                    _MergeMethod(name="interpolate"),
                    None,
                    special_layer.name,
                    special_layer.layer_type.value
                )
                normalized_data.append(embed_slice)

            if special_layer.layer_type == "post_norm" or special_layer.layer_type == "head":
                layer = self._create_slice(
                    unique_sources(get_plain_sources(get_last_block_sources(normalized_data))),
                    special_layer.name,
                    _MergeMethod(name="interpolate"),
                    None,
                    special_layer.name,
                    special_layer.layer_type.value
                )
                normalized_data.append(layer)

        return normalized_data

    def _process_template_slices(self, _slice: _Slice) -> List[_Slice]:
        # should only be passed non-special-layer slices
        # ie. ones that take layer index to be templated and thus are expanded
        # to match the range indicated - or ones where a single 'layer' is
        # already specified manually (below)
        base_model = self._determine_base_model(_slice.sources)
        base_source = self._determine_base_source(_slice.sources)
        layer_name_templates = [
            layer for _, layer in self.models_layers[base_model].items() if layer.layer_type == "decoder"
        ]

        def get_slices_for_all_layers(count: int, _slice: _Slice, layers):
            return [
                _Slice(
                    output_layer_name=lnt.name.format(layer_index=_slice.block_id + i),
                    block_id=_slice.block_id + i,
                    merge_method=_slice.merge_method,
                    sources=[
                        _Source(**{**src.__dict__,
                                   **{"layer": lnt.name.format(layer_index=src.range[0] + i), "range": None}})
                        # src.update("layer", lnt.format(layer_index=src.range[0] + i)).update("range", None)
                        for src in _slice.sources
                    ],
                    layer_type=lnt.layer_type.value,
                )
                for i in range(count)
                for lnt in layers
            ]

        # Range syntax and no filtering
        if all(src.range is not None for src in _slice.sources) and _slice.layers is None:
            start, end = _slice.sources[0].range
            return get_slices_for_all_layers((end - start + 1), _slice, layer_name_templates)

        # Range syntax and layers filtering applied
        elif all(src.range is not None for src in _slice.sources) and _slice.layers is not None:
            # Layers filter applied
            # We create slices with the layers defined by user in the layers filter
            # and then fill in the rest of the layers based on the model architecture definition.

            # Validate if requested layers are at all available in the architecture of the base model
            self._validate_layers_filter_values(base_model, _slice)

            start, end = _slice.sources[0].range
            user_requested_layers = [
                layer
                for requested_layer_type in _slice.layers
                for layer in self.models_layers_by_type[base_model][requested_layer_type]
            ]

            remaining_layers = [l for _, l in self.models_layers[base_model].items() if
                                l not in user_requested_layers and l.layer_type == ModelWeightLayerType.decoder.value]

            user_requested_slices = get_slices_for_all_layers((end - start + 1), _slice, user_requested_layers)
            remaining_slices = [
                _Slice(
                    block_id=_slice.block_id + i,
                    merge_method=_MergeMethod(name="passthrough"),
                    output_layer_name=lnt.name.format(layer_index=_slice.block_id + i),
                    sources=[
                        _Source(model=base_model, is_base=True,
                                layer=lnt.name.format(layer_index=base_source.range[0] + i))
                    ],
                    layer_type=lnt.layer_type.value
                )
                for i in range(end - start + 1)
                for lnt in remaining_layers
            ]

            return user_requested_slices + remaining_slices

        raise Exception("Slice provided without range of layers to merge")

    def _validate_layers_filter_values(self, base_model: str, _slice: _Slice):
        for l in _slice.layers:
            if l not in self.models_layers_by_type[base_model]:
                raise Exception(f"Layer '{l}' does not exist in the model")

    def _create_slice(
            self, sources: List[_Source], layer: Optional[str], merge_method: _MergeMethod, block_id: int | None,
            output_layer_name: str, layer_type: str
    ) -> _Slice:
        # creates a slice, sets merge_method and layer
        # while making sure to keep all other keys.
        # Determines if 'layer' is specified in any source,
        # if not, use the higher-level layer
        sources_with_layer = [
            _Source(**{
                **src.__dict__,
                **{"layer": layer or src.layer},
                **{"weight": src.weight if merge_method.name != "passthrough" else None}
            })
            for src in sources
        ]
        return _Slice(
            output_layer_name=output_layer_name,
            block_id=block_id,
            merge_method=merge_method,
            sources=sources_with_layer,
            layer_type=layer_type,
        )

    def _determine_base_model(self, sources: List[_Source]) -> str | None:
        # determine which model name is the base_model for the given slice
        base_source = self._determine_base_source(sources)
        return base_source.model if base_source is not None else None

    def _determine_base_source(self, sources: List[_Source]) -> _Source:
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
            arch = self.model_arch_provider.get_by_id(m)
            self.models_layers[m] = {weight.name: weight for weight in arch.raw_weights}

            # Group weights in type groups
            self.models_layers_by_type[m] = {
                weight.type: [w for w in arch.raw_weights if w.type is weight.type]
                for weight in arch.raw_weights
            }
