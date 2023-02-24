from __future__ import annotations

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from accelerate import Accelerator
from accelerate.utils.random import set_seed
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.regression import SpearmanCorrCoef
from transformers import AutoTokenizer

from cosent.data import CoSentCollator, CoSentDataset
from cosent.model import CoSentModel
from cosent.trainer import MetricModuleUpdateArgs, MetricStrategy, Trainer, evaluate


class CoSentDatasetType(str, Enum):
    ATEC = 'ATEC'
    BQ = 'BQ'
    LCQMC = 'LCQMC'
    PAWSX = 'PAWSX'
    STSB = 'STS-B'


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

    dataset_name = dataset_type.value
    output_dir = output_dir or Path('experiments') / dataset_name

    set_seed(seed)

    train_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.train.data')
    validation_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.valid.data')
    test_dataset = CoSentDataset(dataset_root_dir / dataset_name / f'{dataset_name}.test.data')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    collator = CoSentCollator(tokenizer, max_length=max_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)

    model = CoSentModel(model_name_or_path, alpha=alpha)
    optimizer = Adam(model.parameters(), lr=lr)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_dir=output_dir,
        log_with=['tensorboard'] if use_tensorboard else None,
    )
    accelerator.init_trackers(dataset_name)

    def metric_adapter(batch_output, batch) -> MetricModuleUpdateArgs:
        return {
            'preds': batch_output['similarities'],
            'target': batch['similarities'],
        }

    metric_strategy = MetricStrategy(
        metric_module=SpearmanCorrCoef(),
        metric_adapter=metric_adapter,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        epochs=epochs,
        metric_strategy=metric_strategy,
    )
    trainer.train()
    model.load_state_dict(trainer.best_model_state_dict)
    test_dataloader = accelerator.prepare(test_dataloader)
    test_metric = evaluate(
        model=model,
        dataloader=test_dataloader,
        metric_strategy=metric_strategy,
    )
    accelerator.log(test_metric)
    print(test_metric)


if __name__ == '__main__':
    typer.run(main)
