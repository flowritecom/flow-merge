from pydantic import BaseModel
from typing import List, Dict, Optional

from flow_merge.lib.logger import Logger

class ModelLayers(BaseModel):
    base_model: Optional[str] = None
    models: Optional[Dict[str, List[str]]] = {}

def get_base_model(slices):
    for slice_entry in slices:
        sources = slice_entry['slice']['sources']
        for source in sources:
            if source.get('base_model', False):
                return source['model']
    return None

def extract_models_by_layers(slices, logger: Logger):
    try:
        if not slices:
            raise TypeError("Empty slices list. Can not resolve layers")
        
        base_model = get_base_model(slices)

        if base_model is None:
            logger.error("No base model identified for the normalized layers.")
            # Log error -> invalid normalization -> invalid snapshot
        
        # Initialize an empty dictionary to store model layers
        models_by_layers = {}

        # Iterate through each slice
        for slice in slices:
            # Get the sources from the slice
            sources = slice['slice']['sources']
            
            # Iterate through each source in the slice
            for source in sources:
                # Check if the source doesn't have base_model set to True
                if not source.get('base_model', False):
                    model = source['model']
                    layer = source['layer']
                    
                    # Initialize the list for the model if not already present
                    if model not in models_by_layers:
                        models_by_layers[model] = set()
                    
                    # Add the layer to the model's set of layers
                    models_by_layers[model].add(layer)

        # Convert sets to lists
        models_by_layers = {model: list(layers) for model, layers in models_by_layers.items()}
        
        return ModelLayers(
            base_model=base_model,
            models=models_by_layers
            )
    except Exception as e:
        raise TypeError(f"Invalid slices format. Exception: {e}")