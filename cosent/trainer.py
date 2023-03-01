from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
import tqdm
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from cosent.trainer_utils import BestModelTracker, MetricModule, MetricTracker


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None,
        accelerator: Accelerator,
        epochs: int,
        lr_scheduler: LRScheduler | None = None,
        metrics: MetricModule | Sequence[MetricModule] = (),
        core_metric_name: str = '-loss',
        log_interval: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval

        if core_metric_name.startswith('-'):
            higher_is_better = False
        elif core_metric_name.startswith('+'):
            higher_is_better = True
        else:
            raise ValueError(f'core_metric_name must start with + or - but got {core_metric_name}')

        self.core_metric_name = core_metric_name[1:]
        self.best_model_tracker = BestModelTracker(higher_is_better)
        self.train_metric_tracker = MetricTracker(deepcopy(metrics))
        self.validation_metric_tracker = MetricTracker(deepcopy(metrics))

    def train(self):
        num_cumulate_batch = 0
        for current_epoch in range(1, self.epochs + 1):

            self.progress_bar = tqdm.tqdm(total=len(self.train_dataloader))
            self.model = self.model.train()
            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.train_metric_tracker.update(batch, batch_output)

                self.progress_bar.update(1)
                num_cumulate_batch += 1

                if batch_index % self.log_interval == 0:
                    metric_scores = self.train_metric_tracker.compute()
                    self.accelerator.log(metric_scores, step=num_cumulate_batch)
                    self._set_description(current_epoch, metric_scores)

            train_metrics = self.add_prefix(self.train_metric_tracker.compute(), 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_metric_tracker.end_epoch()
            self.progress_bar.close()

            if self.validation_dataloader:
                self.model = self.model.eval()
                for batch in self.validation_dataloader:
                    with torch.inference_mode():
                        batch_output = self.model(**batch)
                        self.validation_metric_tracker.update(batch, batch_output)

                validation_metrics = self.validation_metric_tracker.compute()
                self.best_model_tracker.update(
                    model=self.model,
                    score=validation_metrics[self.core_metric_name],
                    epoch=current_epoch,
                )
                validation_metrics = self.add_prefix(validation_metrics, 'validation')
                self.accelerator.log(validation_metrics, step=current_epoch)
                self.validation_metric_tracker.end_epoch()

            self.accelerator.save_state()

        self.accelerator.end_training()

    def _set_description(self, current_epoch: int, metrics: dict[str, float]) -> None:
        description = f'Epoch {current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            description += f' - {name}: {score:.4f}'
        self.progress_bar.set_description(description)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics: MetricModule | Sequence[MetricModule] = (),
):
    metric_tracker = MetricTracker(deepcopy(metrics))

    model = model.eval()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(**batch)
            metric_tracker.update(batch, batch_output)
    metric = metric_tracker.compute()
    return metric
