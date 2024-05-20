from typing import Dict, Optional, Tuple, Type, Union

import torch
from pydantic import ValidationError, field_validator

from flow_merge.lib.architecture import ModelWeight
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_methods.merge_method import (
    BaseMergeMethodSettings,
    MergeMethod,
)
from flow_merge.lib.model import Model

logger = get_logger(__name__)


class TaskArithmeticSettings(BaseMergeMethodSettings):
    scaling_coefficient: Optional[float] = 0.8
    weights: Optional[Dict[Model, float]] = {}

    @field_validator("weights", check_fields=False)
    @classmethod
    def validate_weights(cls, v):
        if v:
            for model, weight in v.items():
                if weight <= 0.0:
                    raise ValidationError(
                        "weights",
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}'"
                        + " from models if you don't want to use the model in the merge.",
                    )
            return v

    @field_validator("scaling_coefficient")
    @classmethod
    def validate_scaling_coefficient(cls, v):
        if v:
            if not 0.0 <= v <= 1.0:
                raise ValidationError(
                    "scaling_coefficient",
                    "Scaling coefficient should be a value between 0.0 and 1.0. It is used to"
                    + "scale the task vectors before adding to the base model tensor. It referred"
                    + " as scaling term in the paper Editing Models with Task Arithmetic (https://arxiv.org/abs/2212.04089)",
                )
            return v


class DareTiesMergingSettings(TaskArithmeticSettings):
    p: Optional[float] = 0.2
    weights: Optional[Dict[Model, float]] = {}

    @field_validator("weights", check_fields=False)
    @classmethod
    def validate_weights(cls, v):
        if v:
            for model, weight in v.items():
                if weight <= 0.0:
                    raise ValidationError(
                        "weights",
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}'"
                        + " from models if you don't want to use the model in the merge.",
                    )
            return v

    @field_validator("p")
    @classmethod
    def validate_p(cls, v):
        if v:
            if not 0.0 <= v <= 1.0:
                raise ValidationError(
                    "p",
                    "p should be between 0.0 and 1.0. It represents the drop rate for random "
                    + "binary mask for each task vector as described in the DARE approach in the "
                    + "paper Language Models are Super Mario: Absorbing Abilities from Homologous "
                    + "Models as a Free Lunch (https://arxiv.org/abs/2311.03099).",
                )
            return v


class TiesMergingSettings(TaskArithmeticSettings):
    top_k: Optional[float] = 0.2
    weights: Optional[Dict[Model, float]] = {}

    @field_validator("weights", check_fields=False)
    @classmethod
    def validate_weights(cls, v):
        if v:
            for model, weight in v.items():
                if weight <= 0.0:
                    raise ValueError(
                        f"Weight for model '{model.path}' must be greater than 0. Remove '{model.path}' "
                        + "from models if you don't want to use the model in the merge."
                    )
            return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v:
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    "top_k should be a value between 0.0 and 1.0. It represents the fraction of top values "
                    + "to keep in the task vectors based on their magnitude as described in the paper Resolving "
                    + "Interference When Merging Models (https://arxiv.org/abs/2306.01708)."
                )
            return v


class TaskArithmetic(MergeMethod):
    def merge(
        self,
        weight: ModelWeight,
        base_model_tensor: torch.Tensor,
        models_tensors: Dict[Model, torch.Tensor],
        merge_method_settings: Union[TaskArithmeticSettings, TiesMergingSettings],
        base_model: Model,
    ) -> torch.Tensor:
        base_tensor_dtype = base_model_tensor.dtype

        task_vectors: Dict[Model, torch.Tensor] = self._get_task_vectors(
            base_model_tensor, models_tensors
        )

        if not task_vectors:
            logger.warning("No task vectors. Returning the base model tensor.")
            return base_model_tensor

        if type(merge_method_settings) == TiesMergingSettings:
            # Ties-merging top-k pruning
            task_vectors = self._topk_pruning(task_vectors, merge_method_settings.top_k)

        if type(merge_method_settings) == DareTiesMergingSettings:
            task_vectors = self._dare_pruning(task_vectors, merge_method_settings.p)

        # _apply_weights(task_vectors, merge_method_settings.weights)
        weighted_task_vectors, weights_tensors = self._prepare_task_vectors(
            task_vectors, merge_method_settings.weights
        )

        if type(merge_method_settings) in [
            TiesMergingSettings,
            DareTiesMergingSettings,
        ]:
            # TIES-merging sign resolution and disjoint merge
            new_task_vector = self._resolve_signs_and_dis_merge(
                weighted_task_vectors=weighted_task_vectors,
                weights_tensors=weights_tensors,
                normalize=merge_method_settings.normalize,
            )
        else:
            # Addition task arithmetic
            new_task_vector = torch.sum(
                weighted_task_vectors, dim=0
            )  # * We do sum only because if all weights are 1.0, they are normalize to be equal
            if merge_method_settings.normalize:
                norm_term = weights_tensors.sum(dim=0)
                norm_term[norm_term == 0] = 1  # Avoid division by zero
                new_task_vector /= norm_term

        # Apply to base model tensor using scaling term as described in the paper Editing Models with Task Arithmetic (https://arxiv.org/abs/2212.04089)
        merged_tensor = (
            base_model_tensor
            + merge_method_settings.scaling_coefficient * new_task_vector
        )

        return merged_tensor.to(dtype=base_tensor_dtype)

    def _get_task_vectors(
        self, base_model_tensor: torch.Tensor, models_tensors: Dict[Model, torch.Tensor]
    ) -> Dict[Model, torch.Tensor]:
        """
        Obtain the task vectors (or deltas) from a pre-trained model tensor and a set of model tensors as described in the paper Editing Models with Task Arithmetic (https://arxiv.org/abs/2212.04089)

        Note:
            If all task vectors are zero (all models are the same), the function returns an empty dictionary.

        Args:
            base_model_tensor: The tensor of the pre-trained base model.
            models_tensors: A dictionary mapping models to their corresponding tensors.

        Returns:
           A dictionary mapping models to their respective task vectors (deltas).
           If all task vectors are zero, an empty dictionary is returned.
        """
        task_vectors: Dict[Model, torch.Tensor] = {}
        all_zero = True
        for model, tensor in models_tensors.items():
            task_vector = tensor - base_model_tensor
            task_vectors[model] = task_vector
            if not torch.allclose(task_vector, torch.zeros_like(task_vector)):
                all_zero = False

        if all_zero:
            return {}
        else:
            return task_vectors

    def _prepare_task_vectors(
        self, task_vectors: Dict[Model, torch.Tensor], weights: Dict[Model, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare task vectors for merging by applying weights to each task vector.

        Args:
            task_vectors: A dictionary where the keys are Model objects
                and the values are torch.Tensor objects representing the task vectors.
            weights: The weights for weighting each task vector.

        Returns:
            The stacked weighted task vectors for each model and the weights tensors.
        """
        weights = [weights[model] for model in task_vectors.keys()]
        stacked_task_vectors = torch.stack(list(task_vectors.values()), dim=0)
        weights_tensors = torch.tensor(
            weights,
            dtype=stacked_task_vectors.dtype,
            device=stacked_task_vectors.device,
        )
        while len(stacked_task_vectors.shape) > len(weights_tensors.shape):
            weights_tensors.unsqueeze_(-1)

        weighted_task_vectors = stacked_task_vectors * weights_tensors

        return weighted_task_vectors, weights_tensors

    def _topk_pruning(
        self, task_vectors: Dict[Model, torch.Tensor], top_k: float
    ) -> Dict[Model, torch.Tensor]:
        """
        Performs top-k pruning on task vectors as described in TIES-MERGING: Resolving Interference When
        Merging Models (https://arxiv.org/abs/2306.01708)

        Args:
            task_vectors: A dictionary where the keys are Model objects
                and the values are torch.Tensor objects representing the task vectors.
            top_k: The fraction of elements to keep in each task vector.

        Returns:
            A dictionary containing the pruned task vectors for each model.
        """
        pruned_task_vectors: Dict[Model, torch.Tensor] = {}

        for model, tensor in task_vectors.items():
            orig_shape = tensor.shape
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # to become a 2D tensor with a single row

            # Get the number of columns and calculate the number of values to keep and prune
            _, n_cols = tensor.shape
            n_values_to_keep = int(top_k * n_cols)

            # Sort the tensor by absolute value and get the indices of the top-k elements
            _, top_k_indices = torch.topk(tensor.abs(), n_values_to_keep, dim=1)

            # Create a mask tensor with True for the top-k elements in each row
            mask = torch.zeros_like(tensor, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)

            final_mask = (
                mask.squeeze() if orig_shape == tensor.squeeze().shape else mask
            )
            tensor = (
                tensor.squeeze() if orig_shape == tensor.squeeze().shape else tensor
            )

            pruned_task_vectors[model] = tensor * final_mask

        return pruned_task_vectors

    def _resolve_signs_and_dis_merge(
        self,
        weighted_task_vectors: torch.Tensor,
        weights_tensors: torch.Tensor,
        sign_consensus_method: str = "mass",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Resolve signs of task vectors and perform disjoint mean merging as described in TIES-MERGING: Resolving Interference When
        Merging Models (https://arxiv.org/abs/2306.01708)

        This function takes a tensor of weighted task vectors and merges them into a single task vector.
        The merging process involves resolving the signs of the task vectors based on the specified sign consensus method
        and then performing a disjoint mean merging.

        The sign consensus method determines how the signs of the task vectors are resolved. Currently, only the "mass"
        method is supported. In the "mass" method, the majority sign of each parameter across all task vectors is used
        as the final sign. Majority is based on the total positive and negative mass. If a task vector's sign does not
        agree with the majority sign for a parameter, that parameter is excluded from the merging process for that task
        vector.

        After resolving the signs, the function performs a disjoint mean merging. It computes the element-wise mean of
        the task vectors, considering only the parameters that have the same sign as the majority sign (different than zero).

        Args:
            weighted_task_vectors: A tensor containing the weighted task vectors.
            weights_tensors: A tensor containing the weights for each task vector.
            sign_consensus_method: The method used for sign consensus. Currently, only "mass" is supported.
            normalize: Whether to normalize the disjoint merge by dividing by the sum of weights for non-zero elements.

        Returns:
            The merged task vector.

        Raises:
            NotImplementedError: If the specified sign consensus method is not implemented.
        """
        if sign_consensus_method == "mass":
            values_sign = torch.sign(weighted_task_vectors)
            mass = weighted_task_vectors.sum(dim=0)

            majority_sign = torch.sign(mass)

            # if sign does not agree with majority sign, set to False and do not merge those parameters
            mask = values_sign == majority_sign

            # final_task_vectors = (stacked_task_vectors * mask).sum(dim=0)
            masked_weighted_task_vectors = weighted_task_vectors * mask
        else:
            raise NotImplementedError(
                f"Sign resolution method '{sign_consensus_method}' is not implemented"
            )

        # Disjoint merge as describe in paper
        merged_task_vector = self._disjoint_merge(
            masked_weighted_task_vectors, weights_tensors, normalize=normalize
        )

        return merged_task_vector

    def _disjoint_merge(
        self,
        masked_weighted_task_vectors: torch.Tensor,
        weights_tensors: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Calculate the disjoint merge of masked task vectors.

        The function calculates the weighted average of a set of vectors while
        ignoring the zero elements.

        The disjoint weighted mean is calculated by multiplying each non-zero element by its corresponding weight,
        summing the weighted elements along each dimension, and then dividing the sum by the sum of weights
        for non-zero elements in that dimension.

        If all the weights are 1.0, it defaults to a simple average.

        If normalize is set to False, the function will return the sum of the weighted non-zero elements without
        averaging based on weights.

        Args:
            masked_weighted_task_vectors: The masked weighted task vectors.
            weights_tensors: The weights for weighting each task vector.
            normalize: Whether to normalize the disjoint merge by dividing by the sum of weights for non-zero elements.

        Returns:
            The disjoint merged task vector.
        """
        # Create a mask tensor where non-zero elements are True and zero elements are False
        mask = masked_weighted_task_vectors != 0

        # Calculate the sum of weighted non-zero elements along the specified dimension
        sum_weighted_non_zeros = (masked_weighted_task_vectors * mask).sum(
            dim=0
        )  # * The mask is used to ignore the zero elements during the sum operation

        if normalize:
            # Calculate the sum of weights for non-zero elements along the specified dimension
            sum_weights_non_zeros = (weights_tensors * mask).sum(dim=0)

            # Calculate the disjoint weighted mean by dividing the sum of weighted non-zero elements
            # by the sum of weights for non-zero elements, with a minimum value of 1 to avoid division by zero
            avg_task_vector = (
                sum_weighted_non_zeros / sum_weights_non_zeros.clamp(min=1)
            )  # * The clamp operation ensures that the minimum value of num_non_zeros is 1 to avoid division by zero
        else:
            avg_task_vector = sum_weighted_non_zeros

        return avg_task_vector

    def _dare_pruning(
        self, task_vectors: Dict[Model, torch.Tensor], p: float
    ) -> Dict[Model, torch.Tensor]:
        """
        Performs drop and rescale pruning on task vectors as described in Language Models are Super Mario:
        Absorbing Abilities from Homologous Models as a Free Lunch (https://arxiv.org/abs/2311.03099)

        Args:
            task_vectors: A dictionary where the keys are Model objects
                and the values are torch.Tensor objects representing the task vectors.
            p: The drop rate for random binary mask for each task vector.
        Returns:
            A dictionary containing the pruned task vectors for each model.
        """
        pruned_task_vectors: Dict[Model, torch.Tensor] = {}

        for model, tensor in task_vectors.items():
            # Create a binary mask tensor with the same shape as the task vector
            mask = torch.bernoulli(
                torch.full_like(input=tensor, fill_value=p, dtype=tensor.dtype)
            )
            masked_tensor = tensor * (
                1 - mask
            )  # * (1 - mask) because elements with 1 are meant to be dropped

            # Rescale remaining by 1 / (1 - p) to maintain
            rescaled_tensor = torch.div(input=masked_tensor, other=1 - p)

            pruned_task_vectors[model] = rescaled_tensor

        return pruned_task_vectors
