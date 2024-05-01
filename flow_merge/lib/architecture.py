import json
import re
from pydantic import BaseModel, Field
from enum import Enum
from importlib.resources import contents, read_text
from typing import Dict, List, Optional

from transformers import PretrainedConfig

import flow_merge.data.architectures

ArchitectureType = Enum(
    "ArchitectureType", ["MistralForCausalLM", "LlamaForCausalLM", "Qwen2ForCausalLM"]
)
ModelType = Enum("ModelType", ["mistral", "llama", "qwen-1.5"])
ModelWeightType = Enum(
    "ModelWeightType",
    [
        "self_attn",
        "mlp",
        "input_layernorm",
        "embed_tokens",
        "norm",
        "lm_head",
        "post_attention_layernorm",
    ],
)
ProjectionType = Enum(
    "ProjectionType",
    ["v_proj", "q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
ModelWeightLayerType = Enum(
    "ModelWeightLayerType", ["decoder", "embedding", "head", "post_norm"]
)

class ModelWeight(BaseModel):
    """
    Contains information about a weight in the model.

    Attributes:
        name: The name of the weight (e.g., 'model.layers.0.self_attn.v_proj.weight').
        type: The type of the weight (e.g., 'self_attn', 'mlp', etc).
        layer_type: The layer of the model (e.g., 'decoder', 'embedding', etc.).
        layer_idx: The index of the layer if applicable.
        projection: The projection of the weight if applicable. For self_attn and mlp only. Default to None.
    """
    name: str
    type: str
    layer_type: str
    layer_idx: Optional[int] = Field(default=None)
    projection: Optional[str] = Field(default=None)


class ModelArchitecture:
    """
    Contains information about the weights of a decoder ONLY architecture and the model type.

    Note:
        - It does not work with encoder models, only decoder.
        - It makes some assumptions about the architecture that might not work with all models.

    Attributes:
        architectures: The list of architectures for the model (e.g., LlamaForCasualLM, MistralForCausalLM, etc).
        layers: The list of layers in the model.
        model_type: The type of the model (e.g., 'llama').
        config: The config object of the transformers model.
    """

    def __init__(self, architectures: List[str], weights: List[ModelWeight], model_type: str, config: PretrainedConfig):
        self.architectures = architectures
        self.weights = weights
        self.model_type = model_type
        self.config = config

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "ModelArchitecture":
        match len(config.architectures):
            case 1:
                for arch in cls._load_all_arch_templates():
                    config_archs = set(config.architectures)
                    template_archs = set(arch["architectures"])
                    if config_archs.intersection(template_archs):
                        return cls._subst(arch, config)
                raise RuntimeError(
                    "Architecture not found in flow-merge supported architectures."
                )
            case x if x > 1:
                raise RuntimeError("More than one architecture in config")
            case _:
                raise RuntimeError("No architecture in config")

    @staticmethod
    def _subst(arch: dict, config: PretrainedConfig) -> "ModelArchitecture":
        model_weights: List[ModelWeight] = []

        for weight in arch["weights"]:
            weight_type = weight["type"]
            projection = weight.get("projection", None)
            if projection:
                if projection not in [value.name for value in ProjectionType]:
                    raise RuntimeError(
                        "Invalid projection type {} in model {} template.".format(
                            projection, arch["model_type"]
                        )
                    )
            if weight_type in [value.name for value in ModelWeightType]:
                if weight["layer_type"] != ModelWeightLayerType.decoder.name:
                    model_weight = ModelWeight(
                        name=weight["name"],
                        type=weight_type,
                        layer_type=weight["layer_type"],
                        layer_idx=None,
                        projection=projection,
                    )
                    model_weights.append(model_weight)
                else:
                    for layer_index in range(config.num_hidden_layers):
                        model_weight = ModelWeight(
                            name=re.sub(
                                r"{layer_index}", str(layer_index), weight["name"]
                            ),
                            type=weight_type,
                            layer_type=weight["layer_type"],
                            layer_idx=layer_index,
                            projection=projection,
                        )
                        model_weights.append(model_weight)
            else:
                raise RuntimeError(
                    "Invalid weight type {} in model {} template.".format(
                        weight_type, arch["model_type"]
                    )
                )

        return ModelArchitecture(
            architectures=arch["architectures"],
            weights=model_weights,
            model_type=arch["model_type"],
            config=config,
        )

    @staticmethod
    def _load_all_arch_templates() -> List[Dict]:
        filepaths = [
            filepath
            for filepath in contents(flow_merge.data.architectures)
            if filepath.lower().endswith(".json")
        ]
        return [
            json.loads(read_text(flow_merge.data.architectures, filepath))
            for filepath in filepaths
        ]

    def get_num_decoder_blocks(self) -> int:
        return self.config.num_hidden_layers

    def get_all_weights(self) -> List[ModelWeight]:
        return self.weights

    def get_embedding_weights(self) -> Optional[ModelWeight]:
        for weight in self.weights:
            if weight.layer_type == ModelWeightLayerType.embedding.name:
                return weight
        return None

    def get_post_norm_weights(self) -> Optional[ModelWeight]:
        for weight in self.weights:
            if weight.layer_type == ModelWeightLayerType.post_norm.name:
                return weight
        return None

    def get_head_weights(self) -> Optional[ModelWeight]:
        for weight in self.weights:
            if weight.layer_type == ModelWeightLayerType.head.name:
                return weight
        return None

    def get_decoder_weights(self) -> List[ModelWeight]:
        decoder_weights: List[ModelWeight] = []
        for weight in self.weights:
            if weight.layer_type == ModelWeightLayerType.decoder.name:
                decoder_weights.append(weight)
        return decoder_weights

