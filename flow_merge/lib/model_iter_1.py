from typing import Dict, List, Optional
from enum import Enum
import re
import json
from importlib.resources import contents, read_text
from pydantic import BaseModel, Field, field_validator
from transformers import PretrainedConfig, AutoConfig
import yaml
from flow_merge.lib.constants import MergeMethodIdentifier
import flow_merge.data.architectures
from flow_merge.lib.logger import get_logger

### Reading utils

def read_yaml(filepath: str) -> Dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)
    


### --- Lightweight model ref --- ###

class ModelRef(BaseModel, frozen=True):
    name_or_path: str = Field(alias="model")
    
### ----------------------------- ###

### --- Original model architecture --- ###
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
    name: str
    type: str
    layer_type: str
    projection: Optional[str] = Field(default=None)

class BaseModelArchitecture:
    
    def __init__(
        self,
        model_ref: ModelRef,
        config: PretrainedConfig,
    ) -> None:
        
        self.model_ref = model_ref
        self.config = config
        arch = self._validate_architecture(candidates=config.architectures)
        try:
            self.num_hidden_layers = getattr(config, arch["num_layers_config_key"])
        except Exception as e:
            raise RuntimeError(
                f"Error getting number of hidden layers from config {config}. {e}"
            )
        self.weights: List[ModelWeight] = self._get_weight_names(arch=arch)
    
    
    def get_layer_name(self, layer_index: int, type: str, layer_type: str) -> str:
        for weight in self.weights:
            if weight.type == type and weight.layer_type == layer_type:
                return weight.name.replace("{layer_index}", str(layer_index))
        raise ValueError(f"No layer found with type '{type}' and layer_type '{layer_type}'")

    def __str__(self) -> str:
        return f"BaseModelArchitecture(model_ref={self.model_ref}, config={self.config}, weights={self.weights})"
    
    def __repr__(self) -> str:
        return f"BaseModelArchitecture(model_ref={repr(self.model_ref)}, config={repr(self.config)}, weights={repr(self.weights)})"
    
    def _get_weight_names(self, arch: Dict) -> List[ModelWeight]:
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
                model_weight = ModelWeight(
                    name=weight["name"],
                    type=weight_type,
                    layer_type=weight["layer_type"],
                    projection=projection,
                )
                model_weights.append(model_weight)
                # if weight["layer_type"] != ModelWeightLayerType.decoder.name:
                #     model_weight = ModelWeight(
                #         name=weight["name"],
                #         type=weight_type,
                #         layer_type=weight["layer_type"],
                #         projection=projection,
                #     )
                #     model_weights.append(model_weight)
                # else:
                #     for layer_index in range(self.num_hidden_layers):
                #         model_weight = ModelWeight(
                #             name=re.sub(
                #                 r"{layer_index}", str(layer_index), weight["name"]
                #             ),
                #             type=weight_type,
                #             layer_type=weight["layer_type"],
                #             projection=projection,
                #         )
                #         model_weights.append(model_weight)
            else:
                raise RuntimeError(
                    "Invalid weight type {} in model {} template.".format(
                        weight_type, arch["model_type"]
                    )
                )
        return model_weights
                
    def _validate_architecture(self, candidates: List[str]) -> Dict[str, str]:
        match len(candidates):
            case 1:
                for arch in self._load_all_json_templates():
                    config_archs = set(candidates)
                    template_archs = set(arch["architectures"])
                    if config_archs.intersection(template_archs):
                        return arch
                raise RuntimeError(
                    "Architecture not found in flow-merge supported architectures."
                )
            case x if x > 1:
                raise RuntimeError("More than one architecture in config")
            case _:
                raise RuntimeError("No architecture in config")
    
    @staticmethod
    def _load_all_json_templates() -> List[Dict]:
        filepaths = [
            filepath
            for filepath in contents(flow_merge.data.architectures)
            if filepath.lower().endswith(".json")
        ]
        return [
            json.loads(read_text(flow_merge.data.architectures, filepath))
            for filepath in filepaths
        ]

### ----------------------------- ###

### --- Definition --- ###

class LayerTypes(str, Enum):
    SELF_ATTENTION = "self_attn"
    MLP = "mlp"
    INPUT_LN = "input_layernorm"
    POST_ATTENTION_LN = "post_attention_layernorm"
    ALL_LAYERS = "all"

class SliceSource(BaseModel):
    model: ModelRef
    range: List[int]
    model_weight: Optional[float] = Field(default=1.0, alias="weight")
    base: Optional[bool] = Field(default=False)
    
    @field_validator("model", mode="before")
    def parse_model_ref(v):
        return ModelRef(model=v)

class SliceMethod(BaseModel):
    name: str
    parameters: Optional[Dict] = Field(default_factory=dict) # ! Validate method values somewhere else???
    
    @field_validator("name", mode="before")
    def validate_method(v):
        if not isinstance(v, MergeMethodIdentifier):
            try:
                v = MergeMethodIdentifier(v)
            except ValueError:
                raise ValueError(f"Invalid method: {v}")
        return v

class Slice(BaseModel):
    sources: List[SliceSource]
    method: SliceMethod
    layers: Optional[List[str]] = Field(default="all")
    
    @field_validator("layers", mode="before")
    def validate_layers(layers):
        values = []
        for layer in layers:
            if not isinstance(layer, LayerTypes):
                try:
                    v = LayerTypes(layer)
                except ValueError:
                    raise ValueError(f"Invalid layer type: {v}")
            values.append(v)
        return values
    
class ArchitectureDefinition(BaseModel):
    slices: List[Slice]

# --------------------- #


    
# class ValidatedInputData(BaseModel):
#     method: Optional[MergeMethodIdentifier] = None
#     method_global_parameters: Optional[MethodGlobalParameters] = None
#     definition: ArchitectureDefinition

    
# class ValidatedSettingsData(BaseModel):
#     tokenizer_settings: Optional[TokenizerSettings] = TokenizerSettings()
#     device: Optional[DeviceIdentifier] = None
#     directory_settings: Optional[DirectorySettings] = DirectorySettings()
#     hf_hub_settings: Optional[HfHubSettings] = HfHubSettings()

if __name__ == "__main__":
    data = read_yaml("/home/admin/flow-merge/examples/testing.yaml")
    
    # ValidatedSettingsData(
    #     tokenizer_settings=data.get("tokenizer_settings"),
    #     device=data.get("device"),
    #     directory_settings=data.get("directory_settings"),
    #     hf_hub_settings=data.get("hf_hub_settings")
    # )
    
    definition = ArchitectureDefinition(slices=data.get("slices"))
    
    # architecture
    model_ref_a = ModelRef(model="Qwen/Qwen1.5-0.5B")
    model_a_config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B")
    model_a_arch = BaseModelArchitecture(model_ref=model_ref_a, config=model_a_config)
    
    model_ref_b = ModelRef(model="mistralai/Mistral-7B-Instruct-v0.2")
    model_b_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model_b_arch = BaseModelArchitecture(model_ref=model_ref_b, config=model_b_config)