from typing import Dict
from enum import Enum

from flow_merge.lib.merge_methods.linear import merge_linear
from flow_merge.lib.merge_methods.merge_method import BaseMergeMethodSettings
from flow_merge.lib.merge_methods.slerp import merge_slerp, SlerpSettings
from flow_merge.lib.merge_methods.task_arithmetic import (
    DareTiesMergingSettings,
    TaskArithmetic,
    TaskArithmeticSettings,
    TiesMergingSettings,
)


class MergeMethodIdentifier(str, Enum):
    ADDITION_TASK_ARITHMETIC = "addition-task-arithmetic"
    TIES_MERGING = "ties-merging"
    SLERP = "slerp"
    DARE_TIES_MERGING = "dare-ties-merging"
    MODEL_SOUP = "model-soup"
    PASSTHROUGH = "passthrough"
    INTERPOLATE = "interpolate"


method_classes: Dict[MergeMethodIdentifier, TaskArithmetic] = {
    MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmetic,
    MergeMethodIdentifier.MODEL_SOUP.value: merge_linear,
    MergeMethodIdentifier.TIES_MERGING.value: TaskArithmetic,
    MergeMethodIdentifier.DARE_TIES_MERGING.value: TaskArithmetic,
    MergeMethodIdentifier.SLERP.value: merge_slerp,
}

method_configs: Dict[MergeMethodIdentifier, BaseMergeMethodSettings] = {
    MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmeticSettings,
    MergeMethodIdentifier.MODEL_SOUP.value: BaseMergeMethodSettings,
    MergeMethodIdentifier.TIES_MERGING.value: TiesMergingSettings,
    MergeMethodIdentifier.DARE_TIES_MERGING.value: DareTiesMergingSettings,
    MergeMethodIdentifier.SLERP.value: SlerpSettings,
}
