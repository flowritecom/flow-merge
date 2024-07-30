try:
    # FIXME: 1. get the PathOrId
    # FIXME: 2. models
    print("Loading ModelSettings")

    merge_top_level_config = MergeConfig(
        cache_dir = "something",         # def
        local_dir = "something",
        output_dir = "something",        # def
        tokenizer_mode = "something",
        tokenizer_interpolation_method = "something",
        base_model = "something",
    )


    # declaratively (yaml):
    SliceValidator.validate(dict(yaml["definition"]))    # this can change/be rewritten to pydantic
    # or "programatically":
    SliceValidator.validate([
        Slice(
            sources = [
                Source(model = "A", layer="model.layers.1"),
                Source(model = "B", layer="model.layers.9")
            ],
            merge_method = MergeMethod(
                name = "slerp"
                params = {
                    "p": 0.4
                }
            )
        ),
    ])

    # normalizer requires ApplicationConfig to load architectures
    # could instead use injected service to do it, assuming other pieces of code also require loading architectures
    # which I assume is the case
    # so could be:
    # arch_loader = ArchitectureLoader(config = application_config)
    normalizer = NormalizationRunner(architecture_loader = arch_loader, logger = logging.logger)
    normalized_slices, num_hidden_layers = normalizer.run(config_dict)


    merge_plan = {
        "created_at": time.now(),
        "created_by": os.hostname(),
        "merge_config": merge_top_level_config,
        "normalized_slices": normalized_slices,
        # "num_hidden_layers": num_hidden_layers,   # very questionable here
        "lib_version": "2.0",
        "sha": sha(merge_top_level_config["base_model"] + merge_top_level_config["tokenizer_settings"] + normalized_slices)
    }


    # print snapshot
    print(json.dumps(merge_plan))

    results = runner.run(merge_plan)

    def run(merge_plan) -> OutputModel:
        self.load_models()
        self.create_tokenizer(merge_plan)
        # ...
        output_model: Tensor[] = []
        # https://github.com/tdrussell/qlora-pipe/blob/main/merge_lora.py
        # https://github.com/tdrussell/qlora-pipe/blob/main/pipeline_model.py
        for slice in merge_plan.slices:
            output_tensor = self.merge_methods[slice.merge_method].merge(...merge_plan.slices)
            output_model.append(output_tensor)


        return self.post_merge_processing(output_model)

    def post_merge_processing(output_model: Model):
        # whatever is needed
