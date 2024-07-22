# docs for Planner

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
