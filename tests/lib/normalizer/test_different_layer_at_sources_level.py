import yaml
from pprint import pprint

from flow_merge.lib.loaders.normalizer import NormalizationRunner


## FIXME: We need to fill the whole layer block 0 with passthrough from base
def test_different_layer_at_sources_level():
    yaml_input = """
    merge_method: slerp
    sources:
      - model: Qwen/Qwen1.5-7B
        layer: model.layers.0.self_attn.q_proj.weight
      - model: Qwen/Qwen1.5-7B-instruct
        layer: model.layers.1.self_attn.q_proj.weight
    """
    expected = [
        {
            "index": 0,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.embed_tokens.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.embed_tokens.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 1,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.q_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.q_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 2,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.norm.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {"layer": "model.norm.weight", "model": "Qwen/Qwen1.5-7B-instruct"},
                ],
            },
        },
        {
            "index": 3,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "lm_head.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {"layer": "lm_head.weight", "model": "Qwen/Qwen1.5-7B-instruct"},
                ],
            },
        },
    ]

    yaml_loaded = yaml.safe_load(yaml_input)
    pprint(yaml_loaded)
    pprint(type(yaml_loaded))
    processed = NormalizationRunner("qwen1_5.json").normalize([yaml_loaded])
    pprint(processed)
    assert processed == expected
