from flow_merge.lib.model_iter_1 import BaseModelArchitecture, ArchitectureDefinition, SliceMethod
from flow_merge.lib.model_iter_1 import ModelWeight, ModelRef, read_yaml
from transformers import AutoConfig
from typing import List
import re

from pydantic import BaseModel

# class ModelArchitectureConstructor:
    
    
#     def __init__(self, architectures: List[BaseModelArchitecture], definition: ArchitectureDefinition) -> None:
#         self.architectures = architectures
#         self.definition = definition

class InputWeight(BaseModel):
    model_ref: ModelRef
    weight: ModelWeight
    tensor_weight: float
    base: bool
    
class OutputWeight(BaseModel):
    weight: ModelWeight
    

def construct_output_architecture(architectures: List[BaseModelArchitecture], definition: ArchitectureDefinition) -> BaseModelArchitecture:
    
    architectures_dict = {arch.model_ref: arch for arch in architectures}
    
    for slice in definition.slices:
        # ! For each slice there is a merge method and there might be layer filters too
        method: SliceMethod = slice.method
        layers = slice.layers
        # ! each slice has more than one source unless `layer-stacking`
        for source in slice.sources:
            # ! For each source, there is a model ref and therefore we need to look up the original architecture
            arch = architectures_dict[source.model]
            # pull weights out for 
            weights = arch.weights
            # ! GET SOURCE WEIGHTS FROM ORIGINAL MODEL
            source_weights: List[ModelWeight] = []
            for idx in range(source.range[0], source.range[1]): # ! A source has a range of weights too that we need to use to select them
                # ! This is inefficient but weights come in a list with weight names from the template
                for w in weights:
                    name = w.name
                    # "model.layers.22.mlp.down_proj.weight"
                    w_idx = re.findall(r'\d+', name)
                    if len(w_idx) > 1:
                        raise RuntimeError(f"Invalid weight name {name}. More than 1No. index.")
                    elif len(w_idx) == 0: # TODO - how to deal with layers outside transformer block
                        continue
                    else:
                        w_idx = int(w_idx[0])
                    if w_idx == idx:
                        source_weights.append(w)
            # ! Base and model weight
            if source.base:
                ... # TODO - What do I do?
            model_weight = source.weight if source.weight else 1.0
            
            
if __name__ == "__main__":
    data = read_yaml("/home/admin/flow-merge/examples/testing.yaml")    
    definition = ArchitectureDefinition(slices=data.get("slices"))
    
    model_ref_a = ModelRef(model="Qwen/Qwen1.5-0.5B")
    model_a_config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B")
    model_a_arch = BaseModelArchitecture(model_ref=model_ref_a, config=model_a_config)
    
    construct_output_architecture(architectures=[model_a_arch], definition=definition)
                    