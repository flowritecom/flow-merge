from typing import Dict, Any


class SliceValidator:
    def validate(self, s: Dict[str, Any]) -> bool:
        if "sources" not in s:
            raise ValueError("Sources not defined for slice")

        # Check that at least model is allowed as base
        if all("base_model" in source and source["base_model"] is False for source in s["sources"]):
            raise ValueError("No valid source found to set as base_model")

        # check for both range and layer at the top level
        if "range" in s and "layer" in s:
            raise ValueError(
                "Cannot have both 'range' and 'layer' at the top level of the slice"
            )

        # check for both range and layer in any of the sources within the same slice
        if any("range" in src for src in s["sources"]) and any("layer" in src for src in s["sources"]):
            raise ValueError(
                "Slice sources have to be defined with either `layer` or `range`, not both"
            )

        if any("range" in src for src in s["sources"]) and not all("range" in src for src in s["sources"]):
            raise ValueError(
                "If used, `range` has to be used for all sources"
            )

        if any("layer" in src for src in s["sources"]) and not all("layer" in src for src in s["sources"]):
            raise ValueError(
                "If used, `layer` has to be used for all sources"
            )

        # Layers filter used with `layer` syntax – invalid, it can only be used with `range` syntax
        if any("layer" in src for src in s["sources"]) and "layers" in s:
            raise ValueError("Layers filter can only be used with `range` definition, not with `layer`")

        if (
                any(isinstance(src["range"], int) for src in s["sources"])
                and not all(isinstance(src["range"], int) for src in s["sources"])
        ):
            raise ValueError("Range for all sources must only be used either with single layer or range of layers")

        if not any(isinstance(src["range"], int) for src in s["sources"]):
            # Check if range length is bigger than 0
            if any("range" in src and src["range"][0] >= src["range"][1] for src in s["sources"]):
                raise ValueError("Provided layers range is not positive")

            # Check if all ranges are of the same length
            if any("range" in src for src in s["sources"]):
                l = s["sources"][0]["range"][1] - s["sources"][0]["range"][0]
                for src in s["sources"]:
                    if src["range"][1] - src["range"][0] is not l:
                        raise ValueError("All `range` must be of the same length")

        if "merge_method" not in s or not isinstance(s["merge_method"], dict) or "name" not in s["merge_method"]:
            raise ValueError("`merge_method` must be a dictionary with `name` field and optional `params` field")

        if "params" in s["merge_method"] and not isinstance(s["merge_method"]["params"], dict):
            raise ValueError("`merge_method.params` must be a dictionary")

        return True
