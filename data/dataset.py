"""
Data pipeline for SISA experiments.

Supports two modes:
1. Local binary tokens (fast, stable) — prepared by prepare_data.py
2. HF streaming (fallback) — slower, can crash with threading bugs

Synthetic: Parity task for state-tracking evaluation.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, DataLoader


class LocalTokenDataset(Dataset):
    """
    Memory-mapped dataset from pre-tokenized binary shards.

    Reads uint16 token files and serves fixed-length (seq_len+1) blocks.
    Map-style (not iterable) — supports shuffling and multi-worker loading.
    """

    def __init__(self, data_dir: str, seq_len: int = 2048, max_tokens: int | None = None):
        self.seq_len = seq_len
        self.block_len = seq_len + 1  # +1 for shifted target
        data_dir = Path(data_dir)

        # Keep shards as individual memmaps — NO concatenation, uses ~0 RAM
        shard_files = sorted(data_dir.glob("shard_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files in {data_dir}")

        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in shard_files]
        self.shard_sizes = [len(s) for s in self.shards]
        self.shard_offsets = []  # cumulative token offset per shard
        cumsum = 0
        for sz in self.shard_sizes:
            self.shard_offsets.append(cumsum)
            cumsum += sz
        total_available = cumsum

        total = min(max_tokens, total_available) if max_tokens else total_available
        self.total_tokens = total
        self.num_blocks = (total - 1) // seq_len
        print(f"  LocalTokenDataset: {total / 1e9:.2f}B tokens, {self.num_blocks:,} blocks from {len(shard_files)} shards (zero-copy mmap)")

    def __len__(self):
        return self.num_blocks

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens starting at global position `start`, spanning shards if needed."""
        result = np.empty(length, dtype=np.uint16)
        written = 0
        # Find starting shard via binary search
        import bisect
        shard_idx = bisect.bisect_right(self.shard_offsets, start) - 1
        local_pos = start - self.shard_offsets[shard_idx]

        while written < length and shard_idx < len(self.shards):
            shard = self.shards[shard_idx]
            available = len(shard) - local_pos
            to_read = min(available, length - written)
            result[written:written + to_read] = shard[local_pos:local_pos + to_read]
            written += to_read
            shard_idx += 1
            local_pos = 0

        return result

    def __getitem__(self, idx):
        start = idx * self.seq_len
        block = self._read_tokens(start, self.block_len).astype(np.int64)
        input_ids = torch.from_numpy(block[:-1].copy())
        labels = torch.from_numpy(block[1:].copy())
        return input_ids, labels


class StreamingTokenDataset(IterableDataset):
    """
    HF streaming fallback — use only if local data is not available.
    WARNING: May crash due to HF datasets threading bug with CUDA backward.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        seq_len: int = 2048,
        split: str = "train",
        max_tokens: int | None = None,
    ):
        self.seq_len = seq_len
        self.max_tokens = max_tokens

        from transformers import AutoTokenizer
        from datasets import load_dataset

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = 0

        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        buffer = []
        tokens_produced = 0
        block_len = self.seq_len + 1

        for doc in self.dataset:
            text = doc.get("text", "")
            if not text:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(token_ids)
            buffer.append(self.tokenizer.eos_token_id)

            while len(buffer) >= block_len:
                block = buffer[:block_len]
                buffer = buffer[block_len:]

                input_ids = torch.tensor(block[:-1], dtype=torch.long)
                labels = torch.tensor(block[1:], dtype=torch.long)
                yield input_ids, labels

                tokens_produced += self.seq_len
                if self.max_tokens and tokens_produced >= self.max_tokens:
                    return


class ParityDataset(IterableDataset):
    """
    Synthetic parity task for state-tracking evaluation.

    Input: random binary sequence [b_1, b_2, ..., b_L]
    Target: running parity at each position — target[i] = XOR(b_1, ..., b_{i+1})
    """

    def __init__(self, seq_len: int = 128, num_samples: int = 10000, seed: int = 42):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_samples):
            bits = torch.randint(0, 2, (self.seq_len,), generator=rng)
            parity = torch.cumsum(bits, dim=0) % 2
            yield bits, parity


def create_dataloader(config, split: str = "train") -> DataLoader:
    data_dir = Path(config.data_cache_dir)

    if (data_dir / "meta.txt").exists():
        # Use local pre-tokenized data
        dataset = LocalTokenDataset(
            data_dir=str(data_dir),
            seq_len=config.seq_len,
            max_tokens=config.max_tokens,
        )
        return DataLoader(
            dataset,
            batch_size=config.micro_batch,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    else:
        # Fallback to streaming
        print("  WARNING: No local data found, using HF streaming (may be slow/unstable)")
        dataset = StreamingTokenDataset(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            seq_len=config.seq_len,
            split=split,
            max_tokens=config.max_tokens,
        )
        return DataLoader(
            dataset,
            batch_size=config.micro_batch,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
