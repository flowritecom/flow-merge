from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

from flow_merge.lib.architecture import ModelArchitecture
from flow_merge.lib.constants import DeviceIdentifier, MergeMethodIdentifier
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_methods import method_classes, method_configs
from flow_merge.lib.merge_methods.merge_method import (
    BaseMergeMethodSettings,
    MergeMethod,
)
from flow_merge.lib.merge_settings import (
    DirectorySettings,
    HfHubSettings,
    MethodGlobalParameters,
    PathOrId,
    RawModelDict,
    TokenizerSettings,
)
from flow_merge.lib.model import Model
from flow_merge.lib.tensor_loader import TensorRepository
from flow_merge.lib.tokenizer import get_merge_tokenizer
from flow_merge.lib.types import TensorIndex

logger = get_logger(__name__)


class MergeConfig(BaseModel):
    """
    This class encapsulates the configuration for the model merge, including the merge method,
    method-specific configuration, base model, models to be merged, tokenizer configuration,
    directory configuration, and Hugging Face Hub configuration.

    Attributes:
        validated_input_data (ValidatedInputData): Validated data from file or variables.
        method (MergeMethod): The merge method to be used.
        method_config (Any): The configuration for the merge method.
        base_model (Model): The base model to be used in the merge process.
        models (List[Model]): The list of models to be merged.
        tokenizer_settings (TokenizerSettings): The settings for the tokenizer.
        directory_settings (DirectorySettings): The directory settings.
        hf_hub_settings (HfHubSettings): The Hugging Face Hub settings.
        device (str): The device to be used for the merge process.
    """

    data: ValidatedInputData
    method: MergeMethod
    method_config: BaseMergeMethodSettings

    base_model: Model
    models: List[Model]


    tokenizer_settings: TokenizerSettings
    directory_settings: DirectorySettings
    hf_hub_settings: HfHubSettings


    device: str
    base_architecture: ModelArchitecture
    non_base_architectures: List[ModelArchitecture]
    tensor_indices: Dict[Model, TensorIndex]
    tensor_loaders: Dict[Model, TensorRepository]

    def __init__(self, data: ValidatedInputData):
        super().__init__(
            data=data,
            method=method_classes[data.method](),
            method_config=self._get_method_config(
                data.method, data.method_global_parameters
            ),
            # independent from above
            tokenizer_settings=data.tokenizer_settings,
            directory_settings=data.directory_settings,
            hf_hub_settings=data.hf_hub_settings,

            # below dependent on dir_settings
            # and validated input data
            models=self.create_models(data),
            base_model=self.create_base_model(data),

            # can be centralized
            device=self.select_device(),
        )

        # Initialize derived attributes
        self.method_config = self._extract_and_set_weights()
        self.base_architecture = ModelArchitecture.from_config(self.base_model.config)
        self.non_base_architectures = [
            ModelArchitecture.from_config(m.config) for m in self.models
        ]
        assert self.validate_architectures()
        self.tensor_indices = self._get_tensor_indices()
        self.tensor_loaders = self._get_tensor_loaders()
        self.tokenizer = get_merge_tokenizer(self)

    @classmethod
    def load(cls, config: Union[str, Dict[str, Any]]) -> "MergeConfig":
        if isinstance(config, str):
            return cls.from_yaml(config)
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise TypeError(
                "Input to load needs to be either a string path to a YAML config file or a dict"
            )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MergeConfig":
        validated_data = ValidatedInputData(**data)
        return MergeConfig(validated_data)

    @classmethod
    def from_yaml(cls, file_path: str) -> "MergeConfig":
        with open(file_path, "r") as file:
            unvalidated_data = yaml.safe_load(file)
        return cls.from_dict(unvalidated_data)

    def _get_tensor_indices(self):
        return {
            model: TensorIndex(str(model.path), self)
            for model in self.models + [self.base_model]
        }

    def _get_tensor_loaders(self):
        return {
            model: TensorRepository(self.tensor_indices[model], self)
            for model in self.models + [self.base_model]
        }

    def _get_method_config(
        self,
        method: MergeMethodIdentifier,
        method_global_parameters: Optional[MethodGlobalParameters],
    ) -> BaseMergeMethodSettings:
        if method_global_parameters is not None:
            return method_configs[method](**method_global_parameters.model_dump())
        else:
            return method_configs[method]()

    def save_config(self):
        path = f"{self.directory_settings.output_dir}/merge_config.yaml"
        with open(path, "w") as file:
            yaml.dump(self.data, file)

    def select_device(self) -> str:
        device = self.data.device
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Switching to CPU.")
            return "cpu"
        return device

    def create_models(self, data: ValidatedInputData) -> List[Model]:
        """
        Create Model instances for the models to be merged, excluding the base model.

        Returns:
            List[Model]: A list of Model instances.
        """
        return [
            Model.from_path(
                Path(model_data.path_or_id),
                directory_settings=self.directory_settings,
            )
            for model_data in data.models
            if model_data.path_or_id != data.base_model
        ]

    def create_base_model(self, data: ValidatedInputData) -> Model:
        """
        Create a Model instance for the base model to be used in the merge process.

        Returns:
            Model: The base model instance.
        """
        base_model_path_or_id = data.base_model or data.models[0].path_or_id
        for model_data in data.models:
            if model_data.path_or_id == base_model_path_or_id:
                return Model.from_path(
                    model_data.path_or_id,
                    directory_settings=self.directory_settings,
                )

        raise ValueError(
            f"Base model '{base_model_path_or_id}' not found in the list of models: "
            + f"{[model.path_or_id for model in data.models]}."
        )

    def _extract_and_set_weights(
        self,
    ) -> Any:
        data: ValidatedInputData = self.data
        base_model: Model = self.base_model
        models: List[Model] = self.models
        method_identifier: str = self.method
        method_config: Any = self.method_config

        weights: Dict[Model, float] = {}
        for model in [base_model] + models:
            for model_data in data.models:
                if model_data.path_or_id == model.path:
                    weight = model_data.weight if model_data.weight else 1.0
                    weights[model] = weight

        method_config.weights = weights
        if method_identifier != MergeMethodIdentifier.SLERP.value:
            if method_identifier in [
                MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value,
                MergeMethodIdentifier.TIES_MERGING.value,
                MergeMethodIdentifier.DARE_TIES_MERGING.value,
            ]:
                if method_config.weights[base_model] != 1.0:
                    logger.warning(
                        f"Merge methods based on task arithmetic use the full base model. "
                        + f"Ignoring the weight for the base model '{base_model.path}'."
                    )
                    method_config.weights[base_model] = 1.0  # Default

        return method_config

    def validate_architectures(self) -> ModelArchitecture:
        """
        Validate that all architectures are the same and therefore compatible.
        """

        if not all(
            set(self.base_architecture.architectures).intersection(set(a.architectures))
            for a in self.non_base_architectures
        ):
            raise RuntimeError(
                "Merging models with different architectures is not supported."
            )

        if not all(
            self.base_architecture.weights == a.weights
            for a in self.non_base_architectures
        ):
            raise RuntimeError(
                "Merging models with different weights is not supported."
            )

        if not all(
            self.base_architecture.model_type == a.model_type
            for a in self.non_base_architectures
        ):
            raise RuntimeError(
                "Merging models with different architectures is not supported."
            )

        return True

    def get_default_values(self) -> Dict[str, Any]:
        """
        Retrieve the default configuration values for the frontend.

        Returns:
            Dict[str, Any]: A dictionary of default values.
        """
        default_values = {
            "method": self.data.method,
            "method_global_parameters": self.data.method_global_parameters.model_dump()
            if self.data.method_global_parameters
            else None,
            "base_model": self.data.base_model,
            "models": [model.model_dump() for model in self.data.models],
            "tokenizer_settings": self.tokenizer_settings.model_dump(),
            "directory_settings": self.directory_settings.model_dump(),
            "hf_hub_settings": self.hf_hub_settings.model_dump(),
            "device": self.data.device,
        }
        return default_values

    class Config:
        frozen = True