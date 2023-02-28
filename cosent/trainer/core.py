from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import tqdm
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from cosent.trainer.metric_strategy import MetricModule, MetricStrategy, MetricAdapter
from cosent.trainer.trackers import BestModelTracker, MetricTracker


def add_prefix(values: dict[str, Any], prefix: str):
    return {f'{prefix}/{k}': v for k, v in values.items()}


@dataclass
class TrainerState:
    epochs: int
    num_batches_per_epoch: int
    loss: float = 0
    current_epoch: int = 0
    current_batch: int = 0
    cumulate_batch: int = 0

    def start_epoch(self):
        self.loss = 0
        self.current_batch = 0
        self.current_epoch += 1
        self.progress_bar = tqdm.tqdm(
            total=self.num_batches_per_epoch, desc=f'Epoch {self.current_epoch}/{self.epochs}', unit='bat'
        )

    def end_epoch(self):
        self.progress_bar.close()

    def advance_batch(self):
        self.current_batch += 1
        self.cumulate_batch += 1
        self.progress_bar.update(1)

    def show_metric(self, metrics: dict[str, float]) -> None:
        description = f'Epoch {self.current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            description += f' - {name}: {score:.4f}'
        self.progress_bar.set_description(description)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None,
        accelerator: Accelerator,
        epochs: int,
        metric_strategy: MetricStrategy | None = None,
        log_interval: int = 50,
        lr_scheduler: LRScheduler | None = None,
    ):
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.metric_log_interval = log_interval
        self.state = TrainerState(epochs=epochs, num_batches_per_epoch=len(train_dataloader))

        self.metric_strategy = metric_strategy or MetricStrategy()
        metric = self.metric_strategy.metric_module
        self.train_metric_tracker = MetricTracker(deepcopy(metric) if metric else None)
        self.validation_metric_tracker = MetricTracker(deepcopy(metric) if metric else None)
        self.best_model_tracker = BestModelTracker()

    def train(self) -> nn.Module:
        for _ in range(self.state.epochs):
            self.state.start_epoch()

            self.model = self.model.train()
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.train_metric_tracker.update_loss(loss.item())
                    self.train_metric_tracker.update_metric(**self.metric_strategy.metric_adapter(batch, batch_output))
                self.state.advance_batch()
                if self.state.current_batch % self.metric_log_interval == 1:
                    metric = self.train_metric_tracker.get_metric()
                    self.accelerator.log(metric, step=self.state.cumulate_batch)
                    self.state.show_metric(metric)
            train_metrics = add_prefix(self.train_metric_tracker.get_metric(), 'train')
            self.accelerator.log(train_metrics, step=self.state.current_epoch)
            self.train_metric_tracker.end_epoch()
            self.state.end_epoch()

            if self.validation_dataloader is None:
                continue

            self.model = self.model.eval()
            for batch in self.validation_dataloader:
                with torch.inference_mode():
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.validation_metric_tracker.update_loss(loss.item())
                    self.validation_metric_tracker.update_metric(**self.metric_strategy.metric_adapter(batch, batch_output))
            
            validation_metrics = self.validation_metric_tracker.get_metric()
            self.best_model_tracker.update(
                model=self.model,
                score=validation_metrics[self.metric_strategy.core_metric_name],
                epoch=self.state.current_epoch,
                higher_is_better=self.metric_strategy.metric_higher_is_better,
            )
            validation_metrics = add_prefix(validation_metrics, 'validation')
            self.accelerator.log(validation_metrics, step=self.state.current_epoch)
            self.validation_metric_tracker.end_epoch()

        self.accelerator.end_training()
        return self.model

    @property
    def best_model_state_dict(self):
        state_dict = self.best_model_tracker.best_model_state_dict
        if state_dict is None:
            raise ValueError('No best model state dict found, please check if you have called `train` method.')
        return state_dict


def evaluate(
    model: nn.Module, dataloader: DataLoader, metric_module: MetricModule | None, metric_adapter: MetricAdapter | None, prefix: str | None = None
):
    if metric_module is not None and metric_adapter is None:
        raise ValueError('`metric_adapter` is required when `metric_module` is provided.')

    metric_tracker = MetricTracker(deepcopy(metric_module) if metric_module else None)

    model = model.eval()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(**batch)
            loss = batch_output['loss']
            metric_tracker.update_loss(loss.item())
            if metric_adapter is not None:
                metric_tracker.update_metric(**metric_adapter(batch, batch_output))
    metric = metric_tracker.get_metric()
    if prefix:
        metric = add_prefix(metric, prefix)
    return metric
