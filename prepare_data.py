#!/usr/bin/env python3
"""
Prepare tokenized data for SISA experiments.

Downloads SlimPajama via streaming, tokenizes with GPT-NeoX tokenizer,
and saves as memory-mapped binary files for fast local training.

Output format: flat binary file of uint16 token IDs (vocab < 65536)
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


def prepare(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id

    print(f"Tokenizer: {args.tokenizer}, vocab_size={tokenizer.vocab_size}, eos_id={eos_id}")
    print(f"Target: {args.num_tokens / 1e9:.1f}B tokens → {out_dir}")
    print(f"Dataset: {args.dataset}")

    ds = load_dataset(args.dataset, split="train", streaming=True)

    # Accumulate tokens into chunks, write periodically
    buffer = []
    total_tokens = 0
    shard_idx = 0
    shard_size = 100_000_000  # 100M tokens per shard (~200MB as uint16)

    for doc in tqdm(ds, desc="Tokenizing", unit=" docs"):
        text = doc.get("text", "")
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        buffer.append(eos_id)

        # Write shard when buffer is large enough
        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
            shard.tofile(shard_path)
            print(f"  Wrote {shard_path.name} ({shard_size / 1e6:.0f}M tokens)")

            buffer = buffer[shard_size:]
            total_tokens += shard_size
            shard_idx += 1

            if total_tokens >= args.num_tokens:
                break

        if total_tokens >= args.num_tokens:
            break

    # Write remaining tokens
    if buffer and total_tokens < args.num_tokens:
        remaining = min(len(buffer), args.num_tokens - total_tokens)
        shard = np.array(buffer[:remaining], dtype=np.uint16)
        shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
        shard.tofile(shard_path)
        total_tokens += remaining
        shard_idx += 1
        print(f"  Wrote {shard_path.name} ({remaining / 1e6:.0f}M tokens)")

    # Write metadata
    meta_path = out_dir / "meta.txt"
    meta_path.write_text(
        f"total_tokens={total_tokens}\n"
        f"num_shards={shard_idx}\n"
        f"vocab_size={tokenizer.vocab_size}\n"
        f"eos_id={eos_id}\n"
        f"dtype=uint16\n"
        f"dataset={args.dataset}\n"
        f"tokenizer={args.tokenizer}\n"
    )

    print(f"\nDone! {total_tokens / 1e9:.2f}B tokens in {shard_idx} shards → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num-tokens", type=int, default=5_500_000_000,
                        help="Number of tokens to prepare (default: 5.5B for Phase 1 + margin)")
    parser.add_argument("--output-dir", default="/data/sisa_tokens")
    args = parser.parse_args()
    prepare(args)
