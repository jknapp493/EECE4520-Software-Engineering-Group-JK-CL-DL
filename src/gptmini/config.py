from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int = 128
    batch_size: int = 8
    epochs: int = 25
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    lr: float = 3e-4
    out_dir: str = "checkpoints"
    use_amp: bool = True
    num_workers: int = 2
    seed: int = 42
    nltk_limit: int | None = None
    c4_limit: int | None = None