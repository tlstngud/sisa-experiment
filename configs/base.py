from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_type: str = "transformer"
    vocab_size: int = 50277  # GPT-NeoX (design doc spec)
    d_model: int = 768
    n_head: int = 12
    n_layer: int = 12
    d_ff: int = 3072
    d_ff_reduced: int = 2748  # SISA / Reduced-MLP Transformer
    d_state: int = 32  # SISA SSM state dim
    max_seq_len: int = 2048

    # Mamba-2 specific
    n_layer_mamba: int = 31
    d_state_mamba: int = 64
    expand_mamba: int = 2
    d_conv_mamba: int = 4

    # Mamba-3 specific (official mamba-ssm implementation)
    n_layer_mamba3: int = 30  # 149.9M params (~152M target)
    d_state_mamba3: int = 64
    expand_mamba3: int = 2

    # Hybrid specific (Mamba front + Attn/SISA back)
    n_mamba_front: int = 20  # 149.7M with 4 attn layers (~152M target)
    n_attn_back: int = 4


@dataclass
class TrainConfig(ModelConfig):
    # Data
    dataset_name: str = "DKYoon/SlimPajama-6B"
    seq_len: int = 2048
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"

    # Optimization
    lr: float = 6e-4
    lr_lambda: float = 6e-5  # 10x smaller for lambda_raw
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Batch — tuned for ~10GB VRAM
    micro_batch: int = 4
    grad_accum: int = 64  # effective ~524K tokens

    # Schedule
    warmup_steps: int = 500
    max_tokens: int = 5_000_000_000  # 5B

    # System
    compile: bool = True
    grad_checkpoint: bool = False
    num_workers: int = 0  # streaming IterableDataset requires 0

    # Logging
    wandb_project: str = "sisa"
    wandb_run_name: str = ""
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Paths
    output_dir: str = "checkpoints"
    data_cache_dir: str = "/data/sisa_tokens"


def get_phase_config(phase: int, model_type: str) -> TrainConfig:
    if phase == 0:
        cfg = TrainConfig(
            model_type=model_type,
            d_model=256,
            n_head=4,
            n_layer=4,
            d_ff=1024,
            d_ff_reduced=920,
            d_state=32,
            n_layer_mamba=8,
            n_layer_mamba3=8,
            d_state_mamba3=64,
            expand_mamba3=2,
            micro_batch=2,
            grad_accum=16,
            max_tokens=50_000_000,  # 50M
            warmup_steps=50,
            lr=1e-3,
            lr_lambda=1e-4,
            log_interval=5,
            eval_interval=100,
            save_interval=500,
            compile=False,
            wandb_run_name=f"phase0_{model_type}",
        )
    elif phase == 1:
        cfg = TrainConfig(
            model_type=model_type,
            d_model=768,
            n_head=12,
            n_layer=12,
            d_ff=3072,
            d_ff_reduced=2748,
            d_state=32,
            n_layer_mamba=31,
            n_layer_mamba3=30,  # 149.9M params (~152M target, official Mamba3)
            d_state_mamba3=64,
            expand_mamba3=2,
            micro_batch=2,
            grad_accum=128,
            max_tokens=5_000_000_000,  # 5B
            warmup_steps=500,
            lr=6e-4,
            lr_lambda=6e-5,
            wandb_run_name=f"phase1_{model_type}",
        )
    elif phase == 2:
        cfg = TrainConfig(
            model_type=model_type,
            d_model=1024,
            n_head=16,
            n_layer=24,
            d_ff=2944,
            d_ff_reduced=2512,
            d_state=32,
            n_layer_mamba=49,
            n_layer_mamba3=48,  # 365.5M params (~369M target, official Mamba3)
            d_state_mamba3=64,
            expand_mamba3=2,
            micro_batch=2,
            grad_accum=128,
            max_tokens=20_000_000_000,  # 20B
            warmup_steps=2000,
            lr=3e-4,
            lr_lambda=3e-5,
            wandb_run_name=f"phase2_{model_type}",
        )
    elif phase == 3:
        # ~50M scale for scaling analysis
        cfg = TrainConfig(
            model_type=model_type,
            d_model=512,
            n_head=8,
            n_layer=6,
            d_ff=2048,
            d_ff_reduced=1832,
            d_state=32,
            n_layer_mamba=14,
            n_layer_mamba3=14,  # ~50M target, official Mamba3
            d_state_mamba3=64,
            expand_mamba3=2,
            micro_batch=4,
            grad_accum=64,
            max_tokens=5_000_000_000,  # 5B
            warmup_steps=500,
            lr=6e-4,
            lr_lambda=6e-5,
            compile=False,
            wandb_run_name=f"phase3_{model_type}",
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return cfg
