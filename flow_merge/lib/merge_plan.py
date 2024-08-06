import datetime
import hashlib
import json
from pathlib import Path
from typing import List, Any, Optional
from pydantic import BaseModel, computed_field, Field

from flow_merge.lib.loaders.normalizer import NormalizationRunner
from flow_merge.lib.merge_config import MergeConfig
from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlice


class MergePlan(BaseModel):
    created_at: datetime.datetime
    base_model: str
    tokenizer_mode: str
    tokenizer_interpolation_method: str
    slices: Optional[List[NormalizedSlice]] = None
    lib_version: str = Field(
        default="0.0.1"  # fixme use actual version from appropriate source
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: MergeConfig, normalization_runner: NormalizationRunner) -> "MergePlan":
        slices, num_hidden_layers = normalization_runner.normalize({
            "base_model": config.base_model,
            "definition": config.definition,
        })
        return cls(
            created_at=datetime.datetime.now(),
            base_model=config.base_model,
            tokenizer_mode=config.tokenizer_mode,
            tokenizer_interpolation_method=config.tokenizer_interpolation_method,
            normalization_runner=normalization_runner,
            slices=slices,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> "MergePlan":
        with open(file_path, "rb") as f:
            parsed = json.load(f)
            return cls(**parsed)

    @computed_field
    @property
    def sha(self) -> str:
        return hashlib.md5(
            {
                "base_model": self.base_model,
                "tokenizer_mode": self.tokenizer_mode,
                "tokenizer_interpolation_method": self.tokenizer_interpolation_method,
                "slices": self.slices,
                "lib_version": self.lib_version,
            }.__str__().encode("utf-8")
        ).hexdigest()

    def to_json(self):
        return self.model_dump_json(indent=4, exclude_none=True)

# class EnrichedSnapshot(Snapshot):
#     models: List[Model]
#     base_model: Model
#     tokenizer: Any = None
#
# class Model(ModelBase, arbitrary_types_allowed=True):
#     id: ModelId
#     path: Path
#     metadata: ModelMetadata
#     file_to_tensor_index: Optional[Dict]
#     shards: List[ShardFile]
#     architecture: ModelArchitecture

# def _merge_sources(self):
#     for slice in self.enriched_snapshot.normalized:
#         merge_method = slice["merge_method"]
#         method_config = self._get_merge_method(merge_method)
#
#         base_model_weight = None
#         models_with_weights = {}
#         for source in slice["sources"]:
#             # if merge_method is passthrough, pass it along to the merged model
#             if source["base_model"]:
#                 base_model_weight = self.enriched_snapshot.base_model.architecture.get_weight(source["layer"])
#                 continue
#
#             path_or_id = source["model"]
#             model = self._get_model_by_id(path_or_id)
#
#             model_weight = model.architecture.get_weight(source["layer"])
#
#             # using the model as key to get the weight
#             models_with_weights[model] = model_weight
#
#         # FIXME: Change this to function
#         self.merger.merge(
#             base_model=self.enriched_snapshot.base_model,
#             task_base_model_weight=base_model_weight,
#             task_models_with_weights=models_with_weights,
#             tokenizer=self.enriched_snapshot.tokenizer,
#             method_config=method_config,
#             sources=slice["sources"]
#         )
