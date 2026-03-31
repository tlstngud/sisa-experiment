#!/usr/bin/env python3
"""
Quick benchmark evaluation for saved checkpoints.
Runs LAMBADA, HellaSwag, PIQA, ARC-Easy, WinoGrande via lm-eval harness.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

from configs import get_phase_config
from models import create_model


def load_checkpoint(model_type: str, phase: int, ckpt_path: str, device: str = "cuda"):
    config = get_phase_config(phase, model_type)
    model = create_model(config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    step = ckpt.get("step", "?")
    tokens = ckpt.get("tokens_seen", "?")
    print(f"  Loaded {model_type} from {ckpt_path}")
    print(f"  Step: {step}, Tokens: {tokens:,}" if isinstance(tokens, int) else f"  Step: {step}")

    return model, config


from lm_eval.api.model import LM as LMBase


class SimpleLM(LMBase):
    """Minimal lm-eval compatible wrapper."""

    def __init__(self, model, tokenizer, device="cuda", max_length=2048):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self._device = device
        self.batch_size = 1

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size_per_gpu(self):
        return self.batch_size

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _logits(self, input_ids):
        input_ids = input_ids[:, -self.max_length:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(input_ids)

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            ctx, cont = req.arguments if hasattr(req, 'arguments') else req
            ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)
            all_ids = (ctx_ids + cont_ids)[-self.max_length:]

            input_ids = torch.tensor([all_ids[:-1]], device=self.device)
            logits = self._logits(input_ids)
            log_probs = F.log_softmax(logits.float(), dim=-1)

            cont_start = len(all_ids) - len(cont_ids) - 1
            cont_ll = sum(
                log_probs[0, cont_start + i, tid].item()
                for i, tid in enumerate(cont_ids)
                if cont_start + i < log_probs.shape[1]
            )

            greedy = all(
                logits[0, cont_start + i].argmax().item() == tid
                for i, tid in enumerate(cont_ids)
                if cont_start + i < logits.shape[1]
            )

            results.append((cont_ll, greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            (text,) = req.arguments if hasattr(req, 'arguments') else req
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                results.append((0.0,))
                continue
            ids = ids[-self.max_length:]
            input_ids = torch.tensor([ids[:-1]], device=self.device)
            target_ids = torch.tensor(ids[1:], device=self.device)

            logits = self._logits(input_ids)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_lps = log_probs[0].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            results.append((token_lps.sum().item(),))
        return results

    def generate_until(self, requests):
        # Return empty strings — we don't support generation, only scoring
        return [""] * len(requests)

    # lm-eval 0.4.x compatibility
    def _model_call(self, inps):
        return self._logits(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["transformer", "mamba2", "mamba3", "sisa"])
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", default=["lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"  Evaluation: {args.model} — Phase {args.phase}")
    print(f"{'='*60}")

    model, config = load_checkpoint(args.model, args.phase, args.ckpt, device)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = SimpleLM(model, tokenizer, device=device)

    try:
        from lm_eval import evaluator, tasks as lm_tasks

        task_manager = lm_tasks.TaskManager()

        results = evaluator.simple_evaluate(
            model=wrapper,
            tasks=args.tasks,
            batch_size=1,
            task_manager=task_manager,
        )

        print(f"\n{'='*60}")
        print(f"  Results: {args.model}")
        print(f"{'='*60}")
        for task_name, task_result in results.get("results", {}).items():
            acc = task_result.get("acc,none", task_result.get("acc_norm,none", "N/A"))
            print(f"  {task_name:20s}: {acc}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"lm-eval error: {e}")
        print("Falling back to manual LAMBADA evaluation...")

        # Manual LAMBADA
        from datasets import load_dataset
        ds = load_dataset("lambada", split="test")

        correct = 0
        total = 0
        for item in ds:
            text = item["text"]
            words = text.rsplit(" ", 1)
            if len(words) < 2:
                continue
            context, last_word = words
            last_word = " " + last_word

            result = wrapper.loglikelihood([(context, last_word)])[0]
            if result[1]:  # greedy match
                correct += 1
            total += 1

            if total % 100 == 0:
                print(f"  LAMBADA: {correct}/{total} = {correct/total:.4f}")

        print(f"\n  LAMBADA final: {correct}/{total} = {correct/total:.4f}")


if __name__ == "__main__":
    main()
