from flow_merge import FlowMergeManager
from flow_merge.lib.loaders.loader import ConfigLoader
from flow_merge.lib.merge_methods.merge_method import MergeMethod
from flow_merge.lib.planners.planner import Planner
from flow_merge.lib.snapshot.merge_snapshot import SnapshotService
from flow_merge.lib.snapshot.data_architecture.snapshot import Snapshot
from flow_merge.lib.snapshot.data_architecture._metadata import SnapshotMetadata, SnapshotHost
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlices, MergeMethodIdentifier, NormalizedSlice, NormalizedSource
from flow_merge.lib.snapshot.data_architecture._settings import (ModelSettings,
    MergeSettings, DirectorySettings, MethodSettings, TokenizerSettings)
from flow_merge.lib.validators._method_settings import MethodGlobalParameters

from pprint import pprint

from flow_merge.lib.validators._model_settings import RawModelDict

from pathlib import Path

merge = FlowMergeManager()
merge.register_service("load", ConfigLoader)
merge.register_service("snap", SnapshotService)
merge.register_service("plan", Planner)

try:
    # load model soup config
    # FIXME: have this bounded to the merge object
    # FIXME: we don't necessarily need it as object here
    loaded_config = merge.load(config="./examples/model_soup.yaml")
    print()
    pprint(loaded_config)
    print()


    # FLOW 1: from file
    [ (load-file ->) objects -> init ]

    # FLOW 2: from notebook / interactive
    # one-click (jupyter / notebook)
    [ (objects [demo]) -> init ]

    flow_merge.abstraction(models=[model1, model2, model3])
    #--> sub-level1 - load
    #--> sub-level2 - plan
    #--> sub-level3 - run

    # time
    #---> IO ---> PROCESS ---> IO

    # ---> IO ---> PROCESS ---> IO ---> result

    #        (info)
    #---> IO ---> PROCESS ---> IO ---> gather result
    #        ---> PROCESS ---> IO
    #        ---> PROCESS ---> IO

    # load "new" YAML (piecewise notation)
    # produce plan
    # execute run
    # write results

    # CLI run
    # yaml -> plan -> run -> results

    # remove registration / module
    # MERGE SPECIFIC

   # https://github.com/EleutherAI/lm-evaluation-harness



    # FIXME: 1. get the PathOrId
    # FIXME: 2. models
    print("Loading ModelSettings")

    merge_top_level_config = MergeConfig(
        cache_dir = "something",         # def
        local_dir = "something",         # def
        output_dir = "something",        # def
        tokenizer_mode = "something",
        tokenizer_interpolation_method = "something",
        base_model = "something",
    )
    SliceValidator


    # class MergeConfiguration:
    #     base_model: str

    #     # -->> validation pydantic classes


    # models = [
    #     RawModelDict(model="Qwen/Qwen2-0.5B"),
    #     RawModelDict(model="Qwen/Qwen2-0.5B-chat"),
    # ]
    # dummy_model_settings = ModelSettings(
    #     base_model = "Qwen/Qwen2-0.5B",
    #     # models = models,
    # )

    # # FIXME: ?
    # print("Loading TokenizerSettings")
    # dummy_tokenizer_settings = TokenizerSettings(
    #     mode="base",
    #     interpolation_method="linear"
    # )

    # print("Loading MethodGlobalParameters")
    # dummy_method_global_parameters = MethodGlobalParameters()

    # print("Loading MethodSettings")
    # dummy_method_settings = MethodSettings(
    #     method="model-soup",
    #     method_global_parameters=dummy_method_global_parameters,
    # )

    # print("Loading DirectorySettings")
    # dummy_directory_settings = DirectorySettings(
    #     cache_dir=Path(".cache").resolve(),
    #     local_dir=Path("./models").resolve(),
    #     output_dir=Path("./output").resolve()
    # )
    # # FIXME: dir settings insists on output_dir existing

    # print("Loading MergeSettings")
    # dummy_merge_settings = MergeSettings(
    #     models=dummy_model_settings,
    #     method=dummy_method_settings,
    #     tokenizer_settings=dummy_tokenizer_settings,
    #     directory_settings=dummy_directory_settings,
    #     sha="sha"
    # )

    # FIRST BLOCK - MERGE SETTINGS
    # model
    # method
    # -> global_parameters
    # tokenizer
    # directory

    # 1. one yaml creates too many files to process it?
    #   -> can we simplify the codebase?

    #





    print("Loading DummyNormalizedSlices")
    dummy_slices = [
        NormalizedSlice(
                        merge_method="model-soup",
                        sources=[
                            NormalizedSource(
                                weight=float(0.5),
                                model="Qwen/Qwen2-0.5B",
                                layer="model.layers.0.self_attn.k_proj.weight",
                                is_base=False,
                            ),
                            NormalizedSource(
                                weight=float(0.5),
                                model="Qwen/Qwen2-0.5B-chat",
                                layer="model.layers.2.self_attn.k_proj.weight",
                                is_base=True,
                            )
                        ]
                    ),
        NormalizedSlice(
                        merge_method="model-soup",
                        sources=[
                            NormalizedSource(
                                weight=float(0.5),
                                model="Qwen/Qwen2-0.5B",
                                layer="model.layers.1.self_attn.k_proj.weight",
                                is_base=True,
                            ),
                            NormalizedSource(
                                weight=float(0.5),
                                model="Qwen/Qwen2-0.5B-chat",
                                layer="model.layers.3.self_attn.k_proj.weight",
                                is_base=False,
                            )
                        ]
                    ),
    ]

    #
    print("Loading NormalizedSlices")
    dummy_normalized_slices = NormalizedSlices(
        slices = dummy_slices,
        sha="sha",
    )

    print("Loading SnapshotHost")
    dummy_snapshot_host = SnapshotHost(
        os = "Linux",
        system_architecture="x86_64"

    )

    print("Loading SnapshotMetadata")
    dummy_snapshot_metadata = SnapshotMetadata(
        created_at="TBD",
        library_version="0.1.0",
        host=dummy_snapshot_host,
    )

    print("Loading Snapshot")
    dummy_snapshot = Snapshot(
        sha="TBD",
        metadata=dummy_snapshot_metadata,
        settings=dummy_merge_settings,
        normalized=dummy_normalized_slices,
        num_hidden_layers = 20
    )


    # snap
    # snapshot = merge.snap()
    # pprint(snapshot)

    # create the plan
    # this needs the FirstLevelSnapshot
    print("Planning!")
    plan = merge.plan(snapshot=dummy_snapshot)

    print("Planning complete, printing the plan")

    pprint(plan)

    # try the dummy run
    # fw.run

    # wrap the stages in exec?

except Exception as e:
    pprint(e)
