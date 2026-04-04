from .transformer import TransformerModel
from .sisa import SISAModel
from .mamba2 import Mamba2Model
from .mamba3 import Mamba3Model
from .hybrid import HybridModel


def create_model(config):
    gc = getattr(config, "grad_checkpoint", False)
    builders = {
        "transformer": lambda: TransformerModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_layer,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            grad_checkpoint=gc,
        ),
        "transformer_reduced": lambda: TransformerModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_layer,
            d_ff=config.d_ff_reduced,
            max_seq_len=config.max_seq_len,
            grad_checkpoint=gc,
        ),
        "sisa": lambda: SISAModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_layer,
            d_ff=config.d_ff_reduced,
            d_state=config.d_state,
            max_seq_len=config.max_seq_len,
            grad_checkpoint=gc,
        ),
        "mamba2": lambda: Mamba2Model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer_mamba,
            d_state=config.d_state_mamba,
            expand=config.expand_mamba,
            d_conv=config.d_conv_mamba,
        ),
        "mamba3": lambda: Mamba3Model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer_mamba3,
            d_state=config.d_state_mamba3,
            expand=config.expand_mamba3,
        ),
        "hybrid_transformer": lambda: HybridModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_mamba=config.n_mamba_front,
            n_attn=config.n_attn_back,
            d_ff=config.d_ff,
            d_state_mamba=config.d_state_mamba,
            expand_mamba=config.expand_mamba,
            d_conv_mamba=config.d_conv_mamba,
            max_seq_len=config.max_seq_len,
            backend='transformer',
        ),
        "hybrid_sisa": lambda: HybridModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_mamba=config.n_mamba_front,
            n_attn=config.n_attn_back,
            d_ff=config.d_ff_reduced,
            d_state_mamba=config.d_state_mamba,
            expand_mamba=config.expand_mamba,
            d_conv_mamba=config.d_conv_mamba,
            d_state_sisa=config.d_state,
            max_seq_len=config.max_seq_len,
            backend='sisa',
        ),
    }
    if config.model_type not in builders:
        raise ValueError(f"Unknown model type: {config.model_type}")
    model = builders[config.model_type]()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{config.model_type}] Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model
