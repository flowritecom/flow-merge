import yaml
from pprint import pprint

from flow_merge.lib.loaders.normalizer import NormalizationRunner


def test_range_at_top_level():
    yaml_input = """
    merge_method: slerp
    sources:
      - model: Qwen/Qwen1.5-7B
      - model: Qwen/Qwen1.5-7B-instruct
    range: [0, 1]
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
                        "layer": "model.layers.0.input_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.input_layernorm.weight",
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
                        "layer": "model.layers.0.self_attn.q_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.q_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
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
                        "layer": "model.layers.0.self_attn.q_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.q_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 4,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.k_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.k_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 5,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.k_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.k_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 6,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.v_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.v_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 7,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.v_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.v_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 8,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.self_attn.o_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.self_attn.o_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 9,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.post_attention_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.post_attention_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 10,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.mlp.gate_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.mlp.gate_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 11,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.mlp.up_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.mlp.up_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 12,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.0.mlp.down_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.0.mlp.down_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 13,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.input_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.input_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 14,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.q_proj.weight",
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
            "index": 15,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.q_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.q_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 16,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.k_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.k_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 17,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.k_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.k_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 18,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.v_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.v_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 19,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.v_proj.bias",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.v_proj.bias",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 20,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.self_attn.o_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.self_attn.o_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 21,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.post_attention_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.post_attention_layernorm.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 22,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.mlp.gate_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.mlp.gate_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 23,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.mlp.up_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.mlp.up_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 24,
            "slice": {
                "merge_method": "slerp",
                "sources": [
                    {
                        "base_model": True,
                        "layer": "model.layers.1.mlp.down_proj.weight",
                        "model": "Qwen/Qwen1.5-7B",
                    },
                    {
                        "layer": "model.layers.1.mlp.down_proj.weight",
                        "model": "Qwen/Qwen1.5-7B-instruct",
                    },
                ],
            },
        },
        {
            "index": 25,
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
            "index": 26,
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
