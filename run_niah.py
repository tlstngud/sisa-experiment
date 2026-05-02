#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) evaluation.

Generate long haystacks with hidden passkey, check if model retrieves it correctly.
"""

import argparse
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

from configs import get_phase_config
from models import create_model


FILLER = (
    "The grass is green. The sky is blue. The sun is bright. The sea is deep. "
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    "Nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
)


def make_sample(tokenizer, context_len, passkey, seed):
    rng = random.Random(seed)
    needle = f"\nThe special passkey is {passkey}. Remember it.\n"
    suffix = "\nThe special passkey is"

    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    # filler to fill context, leave room for needle + suffix
    filler_budget = context_len - len(needle_ids) - len(suffix_ids) - 5
    filler_text = (FILLER * 500)[:max(1, filler_budget * 6)]  # overshoot
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)[:filler_budget]

    # insert needle at random position (middle-ish)
    pos = rng.randint(len(filler_ids) // 4, 3 * len(filler_ids) // 4)
    all_ids = filler_ids[:pos] + needle_ids + filler_ids[pos:] + suffix_ids

    return all_ids[:context_len]


def evaluate_niah(model, tokenizer, device, context_len, n_samples=20):
    correct = 0
    max_seq_len = 2048  # model's training seq_len
    passkeys = [f"{random.randint(10000, 99999)}" for _ in range(n_samples)]

    for i, passkey in enumerate(passkeys):
        input_ids = make_sample(tokenizer, min(context_len, max_seq_len), passkey, seed=i)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_tensor)

        # Decode passkey tokens greedily
        passkey_ids = tokenizer.encode(f" {passkey}", add_special_tokens=False)

        # Get next 6 greedy tokens from last position
        generated = []
        cur_ids = input_tensor
        for _ in range(len(passkey_ids) + 1):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(cur_ids)
            next_id = out[0, -1].argmax().item()
            generated.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if cur_ids.shape[1] >= 2048:
                break

        generated_text = tokenizer.decode(generated)
        if passkey in generated_text:
            correct += 1

    return correct / n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sisa")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phase", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = "cuda"

    config = get_phase_config(args.phase, args.model)
    if args.d_state is not None:
        config.d_state = args.d_state
    if args.d_ff is not None:
        config.d_ff_reduced = args.d_ff

    model = create_model(config)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    print(f"{'='*60}")
    print(f"  NIAH: {args.tag}")
    print(f"{'='*60}")

    random.seed(42)
    results = {}
    for L in args.lengths:
        score = evaluate_niah(model, tokenizer, device, L, n_samples=args.n_samples)
        results[f"L{L}"] = score
        print(f"  L={L:5d}: {score:.3f}")

    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    main()
