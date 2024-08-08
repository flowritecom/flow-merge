# COMMON
import datetime
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, computed_field, Field

from flow_merge.lib.model import Model
from flow_merge.lib.logger import Logger
from flow_merge.lib.config import ApplicationConfig

# INTERPOLATION
import torch
from typing import Dict, List, Tuple, Any
from flow_merge.lib.merge_methods.slerp import SlerpSettings
# from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource, NormalizedSlice

# MERGER
import torch
from typing import Dict

from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSlice, NormalizedSource
from flow_merge.lib.tokenizer import Tokenizer
from flow_merge.lib.model.architecture import ModelWeight
from flow_merge.lib.tensor.loader import TensorRepository
from flow_merge.lib.merge_methods import MergeMethodIdentifier
# from flow_merge.lib.snapshot.data_architecture._normalized_slices import NormalizedSource


# RUNNER
from flow_merge.lib.enriched_snapshot import EnrichedSnapshot
from flow_merge.lib.loaders.normalizer import MergeMethod
from flow_merge.lib.merge_methods import method_classes, method_configs, MergeMethodIdentifier
# from example_slices import example_slices

class InterpolationRunner:

    @staticmethod
    def _map_tensors(
            tensors: Dict[Model, torch.Tensor],
            input_ids_mappings: Dict[Model, Dict[int, int]],
            hidden_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped_tensors = []
        masks = []
        for model, tensor in tensors.items():
            input_ids_map = input_ids_mappings[model]
            mapped_tensor = torch.zeros(
                (len(input_ids_map), hidden_size), dtype=tensor.dtype
            )
            mask = torch.zeros((len(input_ids_map),), dtype=torch.bool)

            for out_id, in_id in input_ids_map.items():
                if in_id >= 0:
                    mapped_tensor[out_id] = tensor[in_id]
                    mask[out_id] = True

            mapped_tensors.append(mapped_tensor)
            masks.append(mask)

        return torch.stack(mapped_tensors), torch.stack(masks)

    @staticmethod
    def _compute_weights(
            sources: List[NormalizedSource],
            method_config
    ):
        weights = [
            source.weight
            if source.weight or not isinstance(method_config.settings, SlerpSettings) else 1.0
            for source in sources
        ]

        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def interpolate(
            cls,
            base_model: Model,
            all_tensors: Dict[Model, torch.Tensor],
            method_config,
            input_ids_mappings: Dict[Model, Dict[int, int]],
            sources: List[NormalizedSource],
            hidden_dim: int
    ):
        mapped_tensors, masks = cls._map_tensors(
            all_tensors,
            input_ids_mappings,
            hidden_dim
        )

        weights = (
            cls._compute_weights(
                sources,
                method_config
            ).unsqueeze(-1).unsqueeze(-1)
        )

        total_weight = (masks.unsqueeze(-1) * weights).sum(dim=0)

        # FIXME confirm correctness of this
        scale = torch.where(total_weight.abs() < 1e-8, torch.tensor(0.0), 1 / total_weight)

        merged_tensor = (mapped_tensors * weights * masks.unsqueeze(-1)).sum(dim=0) * scale
        return merged_tensor.to(dtype=all_tensors[base_model].dtype)


class MergeExecutor:


    # DOING STUFF WITH IT
    # TODO: transform this one from enriched_snapshot into merge plan consuming fn
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

    def run(
            self,
            base_model: Model,
            task_base_model_weight: ModelWeight,
            task_models_with_weights: Dict[Model, ModelWeight],
            tokenizer: Tokenizer,
            method_config,
            sources
    ):


        # GET TENSORS
        base_model_tensor = self._get_base_model_tensor(
            base_model,
            task_base_model_weight
        )

        models_tensors = self._get_all_model_tensors(
            models_with_weights=task_models_with_weights,
        )

        # CONCAT
        all_tensors = {
            base_model: base_model_tensor,
            **models_tensors
        }

        # VALIDATE TENSORS
        hidden_dim = self._validate_tensor_shapes(
            base_model_weight=task_base_model_weight,
            tensors=all_tensors,
            base_model_layer_type=task_base_model_weight.layer_type
        )

        # MAYBE INTERPOLATE
        if tokenizer.input_ids_mappings and method_config.name == MergeMethodIdentifier.INTERPOLATE:
            return InterpolationRunner.interpolate(
                base_model=base_model,
                all_tensors=all_tensors,
                merge_method=method_config,
                input_ids_mappings=tokenizer.input_ids_mappings,
                sources=sources,
                hidden_dim=hidden_dim
            )
        else:
            # OTHERWISE MERGE USING MERGE METHOD
            return method_config.method.merge(
                weight=task_base_model_weight,
                base_model_tensor=base_model_tensor,
                models_tensors=models_tensors,
                merge_method_settings=method_config.settings,
                base_model=base_model
            )


class MergePlan(BaseModel):
    created_at: datetime.datetime = Field(default=datetime.datetime.now())
    base_model: str
    tokenizer_mode: str
    tokenizer_interpolation_method: str
    slices: List[NormalizedSlice]
    lib_version: str

    @classmethod
    def from_config(cls, config: Any) -> "MergePlan":
        return cls(
            created_at=datetime.datetime.now(),
            base_model=config.base_model,
            tokenizer_mode=config.tokenizer_mode,
            tokenizer_interpolation_method=config.tokenizer_interpolation_method,
            slices=config.slices,
            lib_version=config.lib_version,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> "MergePlan":
        with open(file_path, "rb") as f:
            parsed = json.load(f)
            return cls(**parsed)


    @computed_field
    @property
    def sha(self) -> str:
        obj = self.model_dump_json(exclude={"created_at", "sha"}).encode("utf-8")
        return hashlib.md5(obj).hexdigest()


