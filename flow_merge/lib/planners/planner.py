from typing import List, Tuple
from pathlib import Path

from flow_merge.lib.snapshot import Snapshot
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.planners.resolver import extract_models_by_layers, ModelLayers
from flow_merge.lib.model.model import ModelBase, Model
from flow_merge.lib.snapshot.data_architecture.snapshot import Snapshot
from flow_merge.lib.tokenizer import MergeTokenizerService, Tokenizer


# FIXME: takes a model, tokenizer?
class Planner:
    def __init__(self, env, logger, model_class: Model = Model):
        self.env = env
        self.logger = logger
        self.model_class = model_class

    def plan(self, snapshot: Snapshot):
        self.snapshot = snapshot

        # LOAD THE MODELS
        # 1. need the models on disk
        # 2. need the snapshot.normalized
        # 3. need the snapshot.settings.directory_settings
        
        
        (base_model, models) = self._load_models()
        print("_load_models finished!")
        print("Returned to top level: planner.py")

        print("Start EnrichedSnapshot creation")

        print(models)
        print(type(models))
        print(base_model)
        print(type(base_model))
        
        enriched_snapshot = EnrichedSnapshot(
            models=models,
            base_model=base_model,
            sha=snapshot.sha,
            metadata=snapshot.metadata,
            settings=snapshot.settings,
            normalized=snapshot.normalized
        )
        print("Enrichment complete!")

        print("Running _build_merge_tokenizer")
        tokenizer = self._build_merge_tokenizer(enriched_snapshot)
        print("Build merged tokenizer complete!")

        setattr(enriched_snapshot, "tokenizer", tokenizer)

        return enriched_snapshot


    def _load_models(self) -> Tuple[Model, List[Model]]:
        normalized_slices = self.snapshot.normalized.slices
        models_by_layers = extract_models_by_layers(normalized_slices, self.logger)
        print("extracted models by layer")
        print(models_by_layers)

        # Is there an instance when we wouldn't have the base model in normalized slices?

        # Enriched snapshot passing for models?
        base_model = self.model_class.from_path(
            path=models_by_layers.base_model,
            directory_settings=self.snapshot.settings.directory_settings,
            env=self.env,
            logger=self.logger
        )

        models = []

        for model_id_or_path, layers in models_by_layers.models.items():
            models.append(self.model_class.from_layers(
                layers_to_download=layers,
                path=model_id_or_path,
                directory_settings=self.snapshot.settings.directory_settings,
                env=self.env,
                logger=self.logger
            ))

        print("Loaded models!")
        print(models)

        return (base_model, models)
        

    def _build_merge_tokenizer(self, enriched_snapshot: EnrichedSnapshot):
        return MergeTokenizerService(env=self.env, logger=self.logger).get_merge_tokenizer(enriched_snapshot)
