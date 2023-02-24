from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast

import torch
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric


def smartget(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, '__getitem__'):
        return obj[name]
    return default


class MetricModuleUpdateArgs(TypedDict):
    preds: torch.Tensor
    target: torch.Tensor


class MetricAdapter(Protocol):
    def __call__(self, batch_output: Any, batch: Any) -> MetricModuleUpdateArgs:
        ...


class MetricModule(Protocol):
    higher_is_better: bool

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ...

    def compute(self) -> dict[str, float] | float | torch.Tensor:
        ...

    def reset(self) -> None:
        ...


class DefaultMetricAdapter:
    def __call__(self, batch_output: Any, batch: Any) -> MetricModuleUpdateArgs:
        preds = smartget(batch_output, 'logits')
        target = smartget(batch, 'labels')
        if preds is None or target is None:
            raise ValueError(f'Failed Adapt')
        return {
            'preds': preds,
            'target': target,
        }


@dataclass
class MetricStrategy:
    metric_module: MetricModule | None = None
    metric_adapter: MetricAdapter = field(default_factory=DefaultMetricAdapter)
    core_metric_name: str = 'auto'
    metric_higher_is_better: bool = field(default=None)  # type: ignore

    def __post_init__(self):
        if self.core_metric_name == 'auto':
            if self.metric_module is None or isinstance(self.metric_module, MetricCollection):
                self.core_metric_name = 'loss'
            else:
                self.core_metric_name = self.metric_module.__class__.__name__

        if self.metric_adapter is None:
            self.metric_adapter = DefaultMetricAdapter()

        if self.metric_higher_is_better is not None:
            return

        metric_higher_is_better = None
        if self.core_metric_name == 'loss':
            metric_higher_is_better = False
        elif isinstance(self.metric_module, Metric):
            metric_higher_is_better = self.metric_module.higher_is_better
            assert self.core_metric_name == self.metric_module.__class__.__name__
        elif isinstance(self.metric_module, MetricCollection):
            metric_higher_is_better = self.metric_module[self.core_metric_name].higher_is_better   # type: ignore

        if metric_higher_is_better is None:
            raise ValueError(f'Cannot determine whether metric {self.core_metric_name} is higher is better')
        metric_higher_is_better = cast(bool, metric_higher_is_better)
        self.metric_higher_is_better = metric_higher_is_better
