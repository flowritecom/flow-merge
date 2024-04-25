from typing import Dict

from flow_merge.lib.constants import MergeMethodIdentifier
from flow_merge.lib.merge_methods.linear import Linear
from flow_merge.lib.merge_methods.merge_method import BaseMergeMethodSettings
from flow_merge.lib.merge_methods.slerp import Slerp, SlerpSettings
from flow_merge.lib.merge_methods.task_arithmetic import (
    DareTiesMergingSettings,
    TaskArithmetic,
    TaskArithmeticSettings,
    TiesMergingSettings,
)

method_classes: Dict[MergeMethodIdentifier, TaskArithmetic | Linear | Slerp] = {
    MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmetic,
    MergeMethodIdentifier.MODEL_SOUP.value: Linear,
    MergeMethodIdentifier.TIES_MERGING.value: TaskArithmetic,
    MergeMethodIdentifier.DARE_TIES_MERGING.value: TaskArithmetic,
    MergeMethodIdentifier.SLERP.value: Slerp,
}

method_configs: Dict[MergeMethodIdentifier, BaseMergeMethodSettings] = {
    MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value: TaskArithmeticSettings,
    MergeMethodIdentifier.MODEL_SOUP.value: BaseMergeMethodSettings,
    MergeMethodIdentifier.TIES_MERGING.value: TiesMergingSettings,
    MergeMethodIdentifier.DARE_TIES_MERGING.value: DareTiesMergingSettings,
    MergeMethodIdentifier.SLERP.value: SlerpSettings,
}
