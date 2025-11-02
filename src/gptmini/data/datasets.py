import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import nltk
from nltk.corpus import brown, reuters, gutenberg, webtext, inaugural, movie_reviews
from typing import Iterable

class BlockDataset(Dataset):
    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx):
        x = torch.tensor(self.blocks[idx], dtype=torch.long)
        return x, x.clone()


def _texts_from_nltk(limit: int | None) -> list[str]:
    texts = []
    corpora = [reuters, brown, gutenberg, webtext, inaugural, movie_reviews]
    for corpus in corpora:
        ids = corpus.fileids()
        ids = ids if limit is None else ids[:limit]
        for fid in ids:
            try:
                texts.append(" ".join(corpus.words(fid)))
            except Exception:
                continue
    return texts


def _texts_from_small_c4(limit: int | None) -> list[str]:
    ds = load_dataset("brando/small-c4-dataset", split="train")
    texts = []
    for ex in ds:
        t = ex.get("text", "")
        if t:
            texts.append(t)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def build_token_blocks(tokenizer, texts: Iterable[str], block_size: int) -> list[list[int]]:
    blocks: list[list[int]] = []
    for txt in texts:
        ids = tokenizer.encode(txt)
        for i in range(0, len(ids) - block_size, block_size):
            blocks.append(ids[i:i+block_size])
    return blocks


def fix_out_of_range(blocks: list[list[int]], vocab_size: int, eos_id: int) -> list[list[int]]:
    fixed = []
    for b in blocks:
        fixed.append([tok if 0 <= tok < vocab_size else eos_id for tok in b])
    return fixed

class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        # batch is a list of (x, y) tensors
        inputs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return inputs, labels
    
def make_splits(blocks: list[list[int]], batch_size: int, pad_id: int, num_workers: int):
    random.shuffle(blocks)
    n = len(blocks)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)

    collator = PadCollator(pad_id)

    train = DataLoader(
        BlockDataset(blocks[:train_end]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val = DataLoader(
        BlockDataset(blocks[train_end:val_end]),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test = DataLoader(
        BlockDataset(blocks[val_end:]),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return train, val, test

def load_texts(nltk_limit: int | None, c4_limit: int | None) -> list[str]:
    nltk.download("brown"); nltk.download("reuters"); nltk.download("gutenberg")
    nltk.download("webtext"); nltk.download("inaugural"); nltk.download("movie_reviews")
    return _texts_from_nltk(nltk_limit) + _texts_from_small_c4(c4_limit)