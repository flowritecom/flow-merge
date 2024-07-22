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
from typing import Optional, Protocol
from returns.maybe import Maybe, maybe
from returns.context import RequiresContext

from returns.result import Result, safe
from returns.pipeline import flow
from returns.pointfree import bind

# def fetch_user_profile(user_id: int) -> Result['UserProfile', Exception]:
#     """Fetches `UserProfile` TypedDict from foreign API."""
#     return flow(
#         user_id,
#         _make_request,
#         bind(_parse_json),
#     )

# @safe
# def _make_request(user_id: int) -> requests.Response:
#     response = requests.get('/api/users/{0}'.format(user_id))
#     response.raise_for_status()
#     return response

# @safe
# def _parse_json(response: requests.Response) -> 'UserProfile':
#     return response.json()

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


    # FIXME: 1. get the PathOrId
    # FIXME: 2. models
    print("Loading ModelSettings")
    models = [
        RawModelDict(model="Qwen/Qwen2-0.5B"),
        RawModelDict(model="Qwen/Qwen2-0.5B-chat"),
    ]
    dummy_model_settings = ModelSettings(
        base_model = "Qwen/Qwen2-0.5B",
        models = models, 
    )

    # FIXME: ?
    print("Loading TokenizerSettings")
    dummy_tokenizer_settings = TokenizerSettings(
        mode="base",
        interpolation_method="linear"
    )

    print("Loading MethodGlobalParameters")
    dummy_method_global_parameters = MethodGlobalParameters()

    print("Loading MethodSettings")
    dummy_method_settings = MethodSettings(
        method="model-soup",
        method_global_parameters=dummy_method_global_parameters,
    )

    print("Loading DirectorySettings")
    dummy_directory_settings = DirectorySettings(
        cache_dir=Path(".cache").resolve(),
        local_dir=Path("./models").resolve(),
        output_dir=Path("./output").resolve()
    )
    # FIXME: dir settings insists on output_dir existing

    
    # class MergeSettings(BaseModel):
    #     # FIXME We might not have model settings here, with the normalized slices
    #     models: ModelSettings
    #     method: MethodSettings
    #     tokenizer: TokenizerSettings
    #     directory_settings: DirectorySettings
    #     sha: Optional[str]
    print("Loading MergeSettings")
    dummy_merge_settings = MergeSettings(
        models=dummy_model_settings,
        method=dummy_method_settings,
        tokenizer_settings=dummy_tokenizer_settings,
        directory_settings=dummy_directory_settings,
        sha="sha"
    )



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

    print("Loading NormalizedSlices")
    dummy_normalized_slices = NormalizedSlices(
        slices = dummy_slices,
        sha="sha",
    )


    # class SnapshotHost(BaseModel):
    #     os: str
    #     system_architecture: str

    # class SnapshotMetadata(BaseModel):
    #     created_at: str
    #     library_version: str
    #     host: SnapshotHost
    #     sha: Optional[str] = None

    #     @model_validator(mode="after")
    #     def compute_sha(self):
    #         # Convert all fields except 'sha' to a dictionary
    #         data_dict = self.model_dump()
    #         data_dict.pop("sha")
    #         content_hash = create_content_hash(data_dict)
    #         self.sha = content_hash

    #         return self
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

    


