from typing import Any, Dict, List, Optional

import torch
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

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

logger = get_logger(__name__)


class ValidatedInputData(BaseModel):
    method: MergeMethodIdentifier
    method_global_parameters: Optional[MethodGlobalParameters] = None
    base_model: Optional[PathOrId]
    models: List[RawModelDict]
    tokenizer_settings: Optional[TokenizerSettings] = Field(
        TokenizerSettings(), alias="tokenizer"
    )
    directory_settings: Optional[DirectorySettings] = DirectorySettings()
    hf_hub_settings: Optional[HfHubSettings] = HfHubSettings()
    device: Optional[DeviceIdentifier] = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v is not None and v not in ["cpu", "cuda"]:
            raise ValidationError(
                "device",
                f"Invalid device: {v}. Supported devices are 'cpu' and 'cuda'.",
            )
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v) -> None:
        if len(v) < 2:
            raise ValidationError(
                "At least two models should be provided for merging. Either a base model "
                + "and a model or two models if the method provided does not use a base model."
            )
        return v

    @model_validator(mode="after")
    def validate_models_given_method(self) -> Self:
        if len(self.models) > 2 and self.method == MergeMethodIdentifier.SLERP.value:
            raise TypeError("Slerp method requires exactly two models for merging.")
        return self

    @field_validator("method")
    @classmethod
    def validate_merge_method(cls, v):
        if v == MergeMethodIdentifier.PASSTHROUGH.value:
            raise ValidationError("Passthrough merging method is not implemented yet")
        if not v in method_classes:
            valid_methods = [method.value for method in MergeMethodIdentifier]
            raise ValidationError(
                f"Invalid method: '{v}'. Valid methods are: {', '.join(valid_methods)}"
            )
        return v

    @model_validator(mode="after")
    def validate_base_model(self) -> Self:
        base_model = self.base_model
        models = self.models
        if not base_model:
            logger.info(
                f"No base model provided. Merge methods {MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value}",
                f"{MergeMethodIdentifier.TIES_MERGING.value} and {MergeMethodIdentifier.DARE_TIES_MERGING.value}"
                + " use a base model. Using the first model in the list as the base model.",
            )
            try:
                base_model = models[0].model
            except (IndexError, KeyError):
                raise ValueError(
                    "Invalid models list. Please provide a valid list of model dictionaries."
                )
        else:
            base_model_data = next(
                (model for model in models if model.path_or_id == base_model),
                None,
            )
            if not base_model_data:
                raise ValueError(
                    f"Base model '{base_model}' not found in provided models. "
                    + "Please designate one base model from the models."
                )
        return self


class MergeConfig:
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

    def __init__(
        self,
        data: ValidatedInputData,
    ):
        self.data = data
        self.method: MergeMethod = method_classes[data.method]()
        self.method_config: BaseMergeMethodSettings = self._get_method_config(
            data.method, data.method_global_parameters
        )
        self.tokenizer_settings: TokenizerSettings = data.tokenizer_settings
        self.directory_settings: DirectorySettings = data.directory_settings
        self.hf_hub_settings: HfHubSettings = data.hf_hub_settings
        self.device = self.select_device()
        self.models: List[Model] = self.create_models()
        self.base_model: Model = self.create_base_model()
        self.method_config = self._extract_and_set_weights()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MergeConfig":
        validated_data = ValidatedInputData(**data)
        return MergeConfig(validated_data)

    @classmethod
    def from_yaml(cls, file_path: str) -> "MergeConfig":
        with open(file_path, "r") as file:
            unvalidated_data = yaml.safe_load(file)
        return cls.from_dict(unvalidated_data)

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

    def create_models(self) -> List[Model]:
        """
        Create Model instances for the models to be merged, excluding the base model.

        Returns:
            List[Model]: A list of Model instances.
        """
        return [
            Model.from_path(
                model_data.path_or_id,
                directory_settings=self.directory_settings,
            )
            for model_data in self.data.models
            if model_data.path_or_id != self.data.base_model
        ]

    def create_base_model(self) -> Model:
        """
        Create a Model instance for the base model to be used in the merge process.

        Returns:
            Model: The base model instance.
        """
        base_model_path_or_id = self.data.base_model or self.data.models[0].path_or_id
        for model_data in self.data.models:
            if model_data.path_or_id == base_model_path_or_id:
                return Model.from_path(
                    model_data.path_or_id,
                    directory_settings=self.directory_settings,
                )

        raise ValueError(
            f"Base model '{base_model_path_or_id}' not found in the list of models: "
            + f"{[model.path_or_id for model in self.data.models]}."
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
