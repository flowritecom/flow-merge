import json
import re
from enum import Enum
from importlib.resources import contents, read_text
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from transformers import PretrainedConfig

import flow_merge.data.architectures


class ArchitectureType(str, Enum):
    MistralForCausalLM = "MistralForCausalLM"
    LlamaForCausalLM = "LlamaForCausalLM"
    Qwen2ForCausalLM = "Qwen2ForCausalLM"


class ModelType(str, Enum):
    mistral = "mistral"
    llama = "llama"
    qwen_1_5 = "qwen-1.5"


class ModelWeightType(str, Enum):
    self_attn = "self_attn"
    mlp = "mlp"
    input_layernorm = "input_layernorm"
    embed_tokens = "embed_tokens"
    norm = "norm"
    lm_head = "lm_head"
    post_attention_layernorm = "post_attention_layernorm"


class ProjectionType(str, Enum):
    v_proj = "v_proj"
    q_proj = "q_proj"
    k_proj = "k_proj"
    o_proj = "o_proj"
    gate_proj = "gate_proj"
    up_proj = "up_proj"
    down_proj = "down_proj"


class ModelWeightLayerType(str, Enum):
    decoder = "decoder"
    embedding = "embedding"
    head = "head"
    post_norm = "post_norm"


class ModelWeight(BaseModel):
    """
    Contains information about a weight in the model.

    Attributes:
        name: The name of the weight (e.g., 'model.layers.0.self_attn.v_proj.weight').
        type: The type of the weight (e.g., 'self_attn', 'mlp', etc).
        layer_type: The layer of the model (e.g., 'decoder', 'embedding', etc.).
        projection: The projection of the weight if applicable. For self_attn and mlp only. Default to None.
    """

    name: str
    type: ModelWeightType
    layer_type: ModelWeightLayerType
    projection: Optional[ProjectionType] = None


class ModelArchitecture(BaseModel):
    """
    Contains information about the weights of a decoder ONLY architecture and the model type.

    Note:
        - It does not work with encoder models, only decoder.
        - It makes some assumptions about the architecture that might not work with all models.

    Attributes:
        architectures: The list of architectures for the model (e.g., LlamaForCasualLM, MistralForCausalLM, etc).
        weights: The list of weights in the model.
        model_type: The type of the model (e.g., 'llama').
        config: The config object of the transformers model.
    """

    architectures: List[ArchitectureType]
    weights: List[ModelWeight]
    model_type: ModelType
    config: PretrainedConfig

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "ModelArchitecture":
        if len(config.architectures) == 1:
            for arch in cls._load_all_arch_templates():
                config_archs = set(config.architectures)
                template_archs = set(arch["architectures"])
                if config_archs.intersection(template_archs):
                    return cls._subst(arch, config)
            raise RuntimeError(
                "Architecture not found in flow-merge supported architectures."
            )
        elif len(config.architectures) > 1:
            raise RuntimeError("More than one architecture in config")
        else:
            raise RuntimeError("No architecture in config")

    @staticmethod
    def _subst(arch: dict, config: PretrainedConfig) -> "ModelArchitecture":
        model_weights = ModelArchitecture._generate_model_weights(arch, config)
        return ModelArchitecture(
            architectures=arch["architectures"],
            weights=model_weights,
            model_type=arch["model_type"],
            config=config,
        )

    @staticmethod
    def _generate_model_weights(
        arch: dict, config: PretrainedConfig
    ) -> List[ModelWeight]:
        model_weights: List[ModelWeight] = []

        for weight in arch["weights"]:
            weight_type = weight["type"]
            ModelArchitecture._validate_projection(weight, arch)
            if weight_type in [value.name for value in ModelWeightType]:
                if weight["layer_type"] != ModelWeightLayerType.decoder.name:
                    model_weights.append(ModelArchitecture._create_weight(weight))
                else:
                    model_weights.extend(
                        ModelArchitecture._create_decoder_weights(weight, config)
                    )
            else:
                raise RuntimeError(
                    f"Invalid weight type {weight_type} in model {arch['model_type']} template."
                )

        return model_weights

    @staticmethod
    def _create_weight(weight: dict) -> ModelWeight:
        return ModelWeight(
            name=weight["name"],
            type=weight["type"],
            layer_type=weight["layer_type"],
            projection=weight.get("projection", None),
        )

    @staticmethod
    def _create_decoder_weights(
        weight: dict, config: PretrainedConfig
    ) -> List[ModelWeight]:
        decoder_weights: List[ModelWeight] = []
        for layer_index in range(config.num_hidden_layers):
            decoder_weight = ModelWeight(
                name=re.sub(r"{layer_index}", str(layer_index), weight["name"]),
                type=weight["type"],
                layer_type=weight["layer_type"],
                projection=weight.get("projection", None),
            )
            decoder_weights.append(decoder_weight)
        return decoder_weights

    @staticmethod
    def _validate_projection(weight: dict, arch: dict) -> None:
        projection = weight.get("projection", None)
        if projection and projection not in [value.name for value in ProjectionType]:
            raise RuntimeError(
                f"Invalid projection type {projection} in model {arch['model_type']} template."
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
        return next(
            (
                weight
                for weight in self.weights
                if weight.layer_type == ModelWeightLayerType.embedding
            ),
            None,
        )

    def get_post_norm_weights(self) -> Optional[ModelWeight]:
        return next(
            (
                weight
                for weight in self.weights
                if weight.layer_type == ModelWeightLayerType.post_norm
            ),
            None,
        )

    def get_head_weights(self) -> Optional[ModelWeight]:
        return next(
            (
                weight
                for weight in self.weights
                if weight.layer_type == ModelWeightLayerType.head
            ),
            None,
        )

    def get_decoder_weights(self) -> List[ModelWeight]:
        return [
            weight
            for weight in self.weights
            if weight.layer_type == ModelWeightLayerType.decoder
        ]
