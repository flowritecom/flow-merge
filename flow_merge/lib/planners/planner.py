from typing import List

from flow_merge.lib.snapshot import Snapshot
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.planners.resolver import extract_models_by_layers, ModelLayers
from flow_merge.lib.model.model import ModelBase, Model
from flow_merge.lib.snapshot.data_architecture.snapshot import Snapshot
from flow_merge.lib.tokenizer import get_merge_tokenizer, Tokenizer


# Model 
## creates Metadata
## ref to tensor index file
## collects shard files

# We have tokenizer settings at this point
# Create the tokenizer object from the settings

# Previous executor uses tokenizer after executing merge as part of the run 
# (lifecycle continues until successful run and at the point of saving the merged model with the updated tokenizer vocab) implementation:
# # ! If input id mappings rectify the embeeding size, the vocab_size isn't correct in the base model config since it uses the num of embeddings...
# # Update vocab size
# if tokenizer.input_ids_mappings:
#     merged_model_config.vocab_size = len(tokenizer.tokenizer.get_vocab())

# # save tokenizer and config
# logger.info(f"Saving tokenizer to {merge_config.directory_settings.output_dir}")
# tokenizer.tokenizer.save_pretrained(
#     merge_config.directory_settings.output_dir, safe_serialization=True
# )


# Executor steps - previous (for reference):
# 1. Validate model architectures
# 2. Get merge tokenizer
# 3. Build tensor indices and tensor loaders
# 4. Build merger
# 5. Run merge and write tensors
# 6. update tokenizer vocab size -> merged_model_config.vocab_size = len(tokenizer.tokenizer.get_vocab())
# 7. Save tokenizer

### NEW:
# Validate architectures to the front lines -> Entrypoint validation checks
## Planner
# Build models -> create model metadata
# Build merge tokenizer
# update vocab size -> we get the merge_config of the base model or the first model from the models list. Do we already set the first model as the base model in the loading step for the snapshot?
# load tensors

# Snapshot should contain:
# 1. normalized data
# 2. directory settings
# ...


# TODO State sharings:
# As a dependency:
# merge = FlowMerge()
# shared_step_state = GlobalState(id="123")
# merge.inject("state", shared_step_state)
# merge.load()
# merge.plan(optional_state_id="123")

# As a global queue
# Method passing:
# snapshot = merge.load()
# merge.plan(snapshot)


# FIXME: takes a model, tokenizer?
class Planner:
    def __init__(self, env, logger, snapshot: Snapshot, model_class: Model = Model):
        self.env = env
        self.logger = logger
        self.snapshot = snapshot
        self.model_class = model_class

    def plan(self):
        (base_model, models) = self._load_models()
        enriched_snapshot = EnrichedSnapshot(
            **self.snapshot, 
            base_model=base_model, 
            models=models
        )

        tokenizer = self._build_merge_tokenizer(enriched_snapshot)


    def _load_models(self):
        normalized_slices = self.snapshot.normalized

        models_by_layers = extract_models_by_layers(normalized_slices)

        # Is there an instance when we wouldn't have the base model in normalized slices?

        # Enriched snapshot passing for models?
        base_model = self.model_class.from_path(
            path=models_by_layers.base_model,
            directory_settings=self.snapshot.directory_settings,
            env=self.env,
            logger=self.logger
        )

        models = {}

        for model_id_or_path, layers in models_by_layers.models.items():
            models[model_id_or_path] = self.model_class.from_layers(
                layers_to_download=layers,
                path=model_id_or_path,
                directory_settings=self.snapshot.directory_settings,
                env=self.env,
                logger=self.logger
            )

        return (base_model, models)
        

    def _build_merge_tokenizer(self, enriched_snapshot: EnrichedSnapshot):
        return get_merge_tokenizer(enriched_snapshot, self.env, self.logger)
