"""Hyperparameter container for the from-scratch GPT model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50_257       # set to tokenizer.get_vocab_size() after training the BPE tokenizer
    block_size: int = 1024         # max context length (tokens)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True              # GPT-2 style: True. Set False for a slightly faster/leaner variant.

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")


# A much smaller config for the local smoke test (Milestone 1) — trains in
# seconds on CPU so the architecture/training loop can be validated before
# any Colab GPU time is spent.
SMOKE_TEST_CONFIG = ModelConfig(
    vocab_size=256,   # byte-level vocab for the toy smoke test corpus
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=True,
)
