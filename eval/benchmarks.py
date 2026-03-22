"""
LM evaluation harness wrapper for standard benchmarks.

Benchmarks: LAMBADA, HellaSwag, PIQA, ARC-Easy, WinoGrande
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class LMEvalWrapper:
    """
    Wrapper to make our models compatible with lm-eval harness.
    Implements the minimal interface needed for lm_eval.api.model.LM.
    """

    def __init__(self, model, tokenizer_name: str, device: str = "cuda", batch_size: int = 8):
        self.model = model
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def loglikelihood(self, requests):
        """
        Compute log-likelihood of continuations given contexts.
        Each request is (context, continuation).
        """
        results = []

        for context, continuation in requests:
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            all_ids = ctx_ids + cont_ids

            input_ids = torch.tensor([all_ids[:-1]], device=self.device)
            target_ids = torch.tensor([all_ids[1:]], device=self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)

            # Sum log probs over continuation tokens only
            cont_start = len(ctx_ids) - 1
            cont_log_probs = 0.0
            for i, tid in enumerate(cont_ids):
                pos = cont_start + i
                if pos < log_probs.shape[1]:
                    cont_log_probs += log_probs[0, pos, tid].item()

            greedy_match = True
            for i, tid in enumerate(cont_ids):
                pos = cont_start + i
                if pos < logits.shape[1]:
                    if logits[0, pos].argmax().item() != tid:
                        greedy_match = False
                        break

            results.append((cont_log_probs, greedy_match))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood for perplexity."""
        results = []

        for (text,) in requests:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < 2:
                results.append((0.0, 0))
                continue

            input_ids = torch.tensor([token_ids[:-1]], device=self.device)
            target_ids = torch.tensor([token_ids[1:]], device=self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[0].gather(1, target_ids[0].unsqueeze(1)).squeeze(1)
            total = token_log_probs.sum().item()

            results.append((total, len(token_ids) - 1))

        return results


def run_lm_eval(model, tokenizer_name: str, tasks: list[str] | None = None, device: str = "cuda"):
    """
    Run lm-eval benchmarks.

    Default tasks: lambada_openai, hellaswag, piqa, arc_easy, winogrande
    """
    try:
        import lm_eval
        from lm_eval import evaluator
    except ImportError:
        print("lm-eval not installed. Skipping benchmarks.")
        return {}

    if tasks is None:
        tasks = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"]

    wrapper = LMEvalWrapper(model, tokenizer_name, device=device)

    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        batch_size=8,
        device=device,
    )

    # Extract key metrics
    metrics = {}
    for task_name, task_result in results.get("results", {}).items():
        acc = task_result.get("acc,none", task_result.get("acc", None))
        if acc is not None:
            metrics[f"bench/{task_name}_acc"] = acc

    return metrics
