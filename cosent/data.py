from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, TypedDict

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def read_from_csv(csv_file: Path | str, delimiter: str = ',', names: Sequence[str] | None = None):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if names is None:
                names = [f'col_{i}' for i in range(len(row))]
            elif len(row) != len(names):
                print(row)
                raise ValueError(f'row length {len(row)} is not equal to names length {len(names)}')

            yield dict(zip(names, row))


class TextPairSimilarityDict(TypedDict):
    text: str
    text_pair: str
    similarity: float


class CoSentCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[TextPairSimilarityDict]):
        text_ids = self.tokenizer(
            [i['text'] for i in samples], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        )['input_ids']
        text_pair_ids = self.tokenizer(
            [i['text_pair'] for i in samples], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        )['input_ids']
        similarities = torch.tensor([i['similarity'] for i in samples])
        return {'text_ids': text_ids, 'text_pair_ids': text_pair_ids, 'similarities': similarities}


class CoSentDataset(Dataset):
    def __init__(self, data_file: str | Path) -> None:
        self.data_file = Path(data_file)
        self.data = []
        for record in read_from_csv(self.data_file, delimiter='\t', names=['text', 'text_pair', 'similarity']):
            self.data.append(
                TextPairSimilarityDict(
                    text=record['text'], text_pair=record['text_pair'], similarity=float(record['similarity'])
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
