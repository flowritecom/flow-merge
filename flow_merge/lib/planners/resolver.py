from pydantic import BaseModel
from typing import List, Dict, Optional
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlice

from flow_merge.lib.logger import Logger

class ModelLayers(BaseModel):
    base_model: Optional[str] = None
    models: Optional[Dict[str, List[str]]] = {}

def get_base_model(slices: List[NormalizedSlice]):
    try:
        for slice_entry in slices:
            print("accessing slice")
            print(slice_entry)
            sources = slice_entry.sources
            for source in sources:
                print(source)
                if source.is_base:
                    return source.model
        return None
    except Exception as e:
        raise RuntimeError(e)
    

def extract_models_by_layers(slices, logger: Logger):
    print("entering extract models by layers")
    try:
        if not slices:
            raise TypeError("Empty slices list. Can not resolve layers")
        
        print("getting base_model")
        base_model = get_base_model(slices)
        print("got base model")

        if base_model is None:
            logger.error("No base model identified for the normalized layers.")
            # Log error -> invalid normalization -> invalid snapshot
        
        # Initialize an empty dictionary to store model layers
        models_by_layers = {}

        # Iterate through each slice
        for slice in slices:
            # Get the sources from the slice
            print("printing slice")
            print(slice)            
            # Iterate through each source in the slice
            for source in slice.sources:
                # Check if the source doesn't have base_model set to True
                # FIXME: K: Why only non-base?
                if not source.is_base:
                    model = source.model
                    layer = source.layer
                    
                    # Initialize the list for the model if not already present
                    if model not in models_by_layers:
                        models_by_layers[model] = set()
                    
                    # Add the layer to the model's set of layers
                    models_by_layers[model].add(layer)

        # Convert sets to lists
        models_by_layers = {model: list(layers) for model, layers in models_by_layers.items()}

        print("Returning resolver and ModelLayers")
        print(models_by_layers)
        if len(models_by_layers) == 0:
            raise RuntimeError("Models by layers length 0! Check your layer specification!")

        return ModelLayers(
            base_model=base_model,
            models=models_by_layers
            )
    except Exception as e:
        raise TypeError(f"Exception: {e}")
