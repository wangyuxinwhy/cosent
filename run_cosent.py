from __future__ import annotations

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import typer
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.regression import SpearmanCorrCoef
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cosent.data import CoSentCollator, CoSentDataset
from cosent.model import CoSentModel
from cosent.trainer import Trainer, evaluate
from cosent.trainer_utils import MetricModule, MetricModuleUpdateArgs


class CoSentDatasetType(str, Enum):
    ATEC = 'ATEC'
    BQ = 'BQ'
    LCQMC = 'LCQMC'
    PAWSX = 'PAWSX'
    STSB = 'STS-B'


def create_dataloaders(
    dataset_root_dir: Path,
    dataset_name: str,
    tokenizer_name_or_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
):
    train_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.train.data')
    validation_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.valid.data')
    test_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.test.data')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    collator = CoSentCollator(tokenizer, max_length=max_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    return train_dataloader, validation_dataloader, test_dataloader


def create_adamw_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.01):
    parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def main(
    dataset_root_dir: Path,
    dataset_type: CoSentDatasetType,
    model_name_or_path: str,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    seed: int = 42,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    alpha: float = 20,
    lr: float = 2e-5,
    mixed_precision: str = 'bf16',
    gradient_accumulation_steps: int = 1,
    use_tensorboard: bool = False,
):
    set_seed(seed)
    dataset_name = dataset_type.value
    output_dir = output_dir or Path('experiments') / dataset_name

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        automatic_checkpoint_naming=True,
        total_limit=3,
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=['tensorboard'] if use_tensorboard else None,
    )
    accelerator.init_trackers(dataset_name)

    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(
        dataset_root_dir, dataset_name, model_name_or_path, batch_size, max_length, num_workers
    )
    train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, validation_dataloader, test_dataloader
    )
    model = CoSentModel(model_name_or_path, alpha=alpha)
    optimizer = create_adamw_optimizer(model, lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps
    )

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    def adapt_fn(batch, batch_output) -> MetricModuleUpdateArgs:
        return {
            'preds': batch_output['similarities'],
            'target': batch['similarities'],
        }

    metric = MetricModule(
        module=SpearmanCorrCoef(),
        adapt_fn=adapt_fn,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        epochs=epochs,
        metrics=metric,
        core_metric_name='+SpearmanCorrCoef',
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

    model.load_state_dict(trainer.best_model_tracker.best_model_state_dict)
    test_metric = evaluate(
        model=model,
        dataloader=test_dataloader,
        metrics=metric,
    )
    accelerator.log(test_metric)
    print(test_metric)


if __name__ == '__main__':
    typer.run(main)
