from typing import Dict, Any


class SliceValidator:
    def validate(self, s: Dict[str, Any]):
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
