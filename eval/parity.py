"""
Parity task evaluation.

Tests state-tracking ability: given a binary sequence, predict running parity.
SSM-based models (SISA, Mamba-2) should outperform pure Transformers.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import ParityDataset


@torch.no_grad()
def evaluate_parity(
    model,
    seq_len: int = 128,
    num_samples: int = 2000,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    """
    Evaluate model on the parity task.

    The model must map token sequences to token predictions.
    We feed binary sequences (token IDs 0/1) and check if the model
    predicts the correct running parity at each position.
    """
    model.eval()

    dataset = ParityDataset(seq_len=seq_len, num_samples=num_samples, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    total_correct = 0
    total_tokens = 0
    total_correct_last = 0
    total_samples = 0

    for bits, parity in loader:
        bits = bits.to(device)
        parity = parity.to(device)

        logits = model(bits)  # (B, L, vocab_size)
        preds = logits[:, :, :2].argmax(dim=-1)  # only look at tokens 0/1

        correct = (preds == parity).float()
        total_correct += correct.sum().item()
        total_tokens += correct.numel()

        # Accuracy at the last position (hardest)
        total_correct_last += correct[:, -1].sum().item()
        total_samples += correct.shape[0]

    model.train()

    return {
        "parity/token_acc": total_correct / max(total_tokens, 1),
        "parity/last_pos_acc": total_correct_last / max(total_samples, 1),
    }
