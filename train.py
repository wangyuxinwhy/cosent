from enum import Enum
from pathlib import Path
from typing import Optional

import tqdm
import typer


class CoSentDatasetType(str, Enum):
    ATEC = 'ATEC'
    BQ = 'BQ'
    LCQMC = 'LCQMC'
    PAWSX = 'PAWSX'
    STSB = 'STS-B'


def build_progress_bar(current_epoch: int, epochs: int, total_batches: int):
    progress_bar = tqdm.tqdm(total=total_batches, desc='Epoch {current_epoch}/{epochs}', unit='bat')
    return progress_bar


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
):
    import os

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from accelerate import Accelerator
    from accelerate.utils.random import set_seed
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    import tqdm
    from transformers import AutoTokenizer

    from cosent.data import CoSentCollator, CoSentDataset
    from cosent.model import CoSentModel

    dataset_name = dataset_type.value
    output_dir = output_dir or Path.cwd() / 'runs' / dataset_name

    set_seed(seed)

    dataset_name = dataset_type.value
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
    )

    model, optimizer, train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, test_dataloader
    )
    model.train()

    for current_epoch in range(epochs):
        avg_loss = 0
        epoch_description = f'Epoch {current_epoch+1}/{epochs}'
        progress_bar = tqdm.tqdm(total=len(train_dataloader), unit='bat')
        for num_batches, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                model_output = model(**batch)
                loss = model_output['loss']
                accelerator.backward(loss)
                optimizer.step()
                avg_loss = (avg_loss * num_batches + loss.item()) / (num_batches + 1)
            progress_bar.update(1)
            if num_batches % 10 == 0:
                progress_bar.set_description(epoch_description + f' - loss: {loss:.4f}')

    accelerator.end_training()


if __name__ == '__main__':
    typer.run(main)
