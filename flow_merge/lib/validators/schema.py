from typing import List, Optional
from pydantic import BaseModel, field_validator



    # FIXME: This is about legality -> postpone this to normalization
    # @model_validator(mode="after")
    # def validate_models_given_method(self) -> Self:
    #     if len(self.models) > 2 and self.method == MergeMethodIdentifier.SLERP.value:
    #         raise TypeError("Slerp method requires exactly two models for merging.")
    #     return self


    # FIXME: this is also about normalization -> defer?
    # @model_validator(mode="after")
    # def validate_base_model(self) -> Self:
    #     base_model = self.base_model
    #     models = self.models
    #     if not base_model:
    #         logger.info(
    #             f"No base model provided. Merge methods {MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value}",
    #             f"{MergeMethodIdentifier.TIES_MERGING.value} and {MergeMethodIdentifier.DARE_TIES_MERGING.value}"
    #             + " use a base model. Using the first model in the list as the base model.",
    #         )
    #         try:
    #             base_model = models[0].model
    #         except (IndexError, KeyError):
    #             raise ValueError(
    #                 "Invalid models list. Please provide a valid list of model dictionaries."
    #             )
    #     else:
    #         base_model_data = next(
    #             (model for model in models if model.path_or_id == base_model),
    #             None,
    #         )
    #         if not base_model_data:
    #             raise ValueError(
    #                 f"Base model '{base_model}' not found in provided models. "
    #                 + "Please designate one base model from the models."
    #             )
    #     return self
    # class Config:
    #     frozen = True  # Make this model immutable
