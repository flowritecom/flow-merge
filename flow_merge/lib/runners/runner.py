from flow_merge.lib.config import ApplicationConfig
from flow_merge.lib.logger import Logger
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.merger import Merger
from flow_merge.lib.loaders.normalizer import MergeMethod
from flow_merge.lib.merge_methods import method_classes, method_configs, MergeMethodIdentifier

slices = [
    # output_layer_id is per decoder block
    # FIXME base_model is now is_base: Boolean
    {'output_later_id': 0, 'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.embed_tokens.weight', 'model': 'model_1', 'weight': 1.0}]},
    {'output_later_id': 7, 'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_1'}, {'layer': 'model.layers.0.mlp.gate_proj.weight', 'model': 'model_2'}]},
    {'output_later_id': 28, 'merge_method': 'passthrough', 'sources': [{'base_model': True, 'layer': 'model.norm.weight', 'model': 'model_1'}, {'layer': 'model.norm.weight', 'model': 'model_2'}]},
    # Add more entries as needed
]

class Runner:
    def __init__(
            self, 
            enriched_snapshot: EnrichedSnapshot, 
            env: ApplicationConfig, 
            logger: Logger,
            # FIXME Have a base merger here
            merger: Merger = Merger
        ):
        self.env = env
        self.logger = logger
        self.enriched_snapshot = enriched_snapshot
        self.merger = merger

    def _get_model_by_id(self, path_or_id):
        models = self.enriched_snapshot.models
        for model in models:
            if model.id == path_or_id:
                return model
        return None

    def _get_merge_method(self, merge_method: MergeMethod):
        
        if (merge_method.name == MergeMethodIdentifier.INTERPOLATE):
            return {
                "name": merge_method.name,
                "method": None,
                "settings": None
            }

        method_class = method_classes[merge_method.name]
        method_config = method_configs[merge_method.name]

        try:
            settings = method_config(**merge_method.params)
        except Exception as e:
            raise RuntimeError(
                f"Can not instantiate merge method {merge_method.merge_method} parameters"
            )
        
        method_config = {
            "name": merge_method.name,
            "method": method_class,
            "settings": settings
        }

        return method_config
    
    def _get_merged_config(self):
        merged_model_config = self.enriched_snapshot.base_model.architecture.config
        merged_model_config.num_hidden_layers = self.enriched_snapshot.num_hidden_layers

        if self.enriched_snapshot.tokenizer.input_ids_mappings:
            merged_model_config.vocab_size = len(
                self.enriched_snapshot.tokenizer.tokenizer.get_vocab()
            )

        return merged_model_config
    
    def _merge_sources(self):
        for slice in self.enriched_snapshot.normalized:
            merge_method = slice["merge_method"]
            method_config = self._get_merge_method(merge_method)

            base_model_weight = None
            models_with_weights = {}
            for source in slice["sources"]:
                # if merge_method is passthrough, pass it along to the merged model
                if source["base_model"]:
                    base_model_weight = self.enriched_snapshot.base_model.architecture.get_weight(source["layer"])
                    continue

                path_or_id = source["model"]
                model = self._get_model_by_id(path_or_id)

                model_weight = model.architecture.get_weight(source["layer"])
                models_with_weights[model] = model_weight

            self.merger.merge(
                base_model=self.enriched_snapshot.base_model,
                task_base_model_weight=base_model_weight,
                task_models_with_weights=models_with_weights,
                tokenizer=self.enriched_snapshot.tokenizer,
                method_config=method_config,
                sources=slice["sources"]
            )
        
    def run(self):
        self._merge_sources()
        # Pass along to 'save'
        merged_model_config = self._get_merged_config()
                

