from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence, TypedDict, cast

import torch
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric


class MetricModuleUpdateArgs(TypedDict):
    preds: torch.Tensor
    target: torch.Tensor


class MetricAdapter(Protocol):
    def __call__(self, batch: Any, batch_output: Any) -> MetricModuleUpdateArgs:
        ...


@dataclass
class MetricModule:
    module: Metric | MetricCollection
    adapt_fn: MetricAdapter


class MetricTracker:
    def __init__(self, metric_modules: MetricModule | Sequence[MetricModule] = (), ndigits=4) -> None:
        self.metric_modules = (metric_modules,) if isinstance(metric_modules, MetricModule) else metric_modules
        self.ndigits = ndigits
        self.loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[dict[str, float]] = []

    def update(self, batch: Any, batch_output: Any):
        loss = batch_output['loss'].item()
        self.loss = (self.loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1
        for metric_module in self.metric_modules:
            args = metric_module.adapt_fn(batch_output, batch)
            metric_module.module(**args)

    def reset(self):
        self.loss = 0
        self.loss_count = 0
        for metric_module in self.metric_modules:
            metric_module.module.reset()

    def end_epoch(self, reset: bool = True):
        self.history.append(self.compute())
        if reset:
            self.reset()

    def compute(self) -> dict[str, float]:
        output_dict = {'loss': self.loss}

        for metric_module in self.metric_modules:
            score = metric_module.module.compute()
            if not isinstance(score, Mapping):
                score = float(score)
                output_dict[metric_module.module.__class__.__name__] = score
            else:
                score = cast(Dict[str, float], score)
                output_dict.update(score)
        output_dict = {k: round(float(v), self.ndigits) for k, v in output_dict.items()}
        return output_dict


class BestModelTracker:
    def __init__(self, higher_is_better: bool) -> None:
        self.higher_is_better = higher_is_better
        self.best_model_state_dict = None
        self.best_score = None
        self.best_epoch = None

    def update(self, model: torch.nn.Module, score: float, epoch: int) -> None:
        if self.best_score is None:
            self.best_model_state_dict = model.state_dict()
            self.best_score = score
            self.best_epoch = epoch
            return

        if self.higher_is_better:
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score

        if is_better:
            self.best_model_state_dict = model.state_dict()
            self.best_score = score
            self.best_epoch = epoch
