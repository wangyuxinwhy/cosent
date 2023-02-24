from __future__ import annotations

from typing import Dict, Mapping, cast

import torch
import torch.nn as nn

from cosent.trainer.metric_strategy import MetricModule


class MetricTracker:
    def __init__(self, metric_module: MetricModule | None) -> None:
        self.metric_module = metric_module
        self.loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[dict[str, float]] = []

    def update_loss(self, loss: float):
        self.loss = (self.loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def update_metric(self, preds: torch.Tensor, target: torch.Tensor):
        if self.metric_module:
            self.metric_module.update(preds, target)

    def reset(self):
        self.loss = 0
        self.loss_count = 0
        if self.metric_module:
            self.metric_module.reset()

    def end_epoch(self):
        self.history.append(self.get_metric())
        self.reset()

    def get_metric(self) -> dict[str, float]:
        output_dict = {'loss': self.loss}

        if self.metric_module:
            score = self.metric_module.compute()
            if not isinstance(score, Mapping):
                score = float(score)
                output_dict[self.metric_module.__class__.__name__] = score
            else:
                score = cast(Dict[str, float], score)
                output_dict.update(score)

        return output_dict


class BestModelTracker:
    def __init__(self) -> None:
        self.best_model_state_dict = None
        self.best_score = None
        self.best_epoch = None

    def update(self, model: nn.Module, score: float, epoch: int, higher_is_better: bool) -> None:
        if self.best_score is None:
            self.best_model_state_dict = model.state_dict()
            self.best_score = score
            self.best_epoch = epoch
            return

        if higher_is_better:
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score

        if is_better:
            self.best_model_state_dict = model.state_dict()
            self.best_score = score
            self.best_epoch = epoch

    def get_best_model(self, model: nn.Module) -> nn.Module:
        if self.best_model_state_dict is None:
            raise ValueError('No best model state dict found')
        model.load_state_dict(self.best_model_state_dict)
        return model
