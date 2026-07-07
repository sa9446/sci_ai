"""Generates a tiny byte-level train.bin/val.bin for the Milestone-1 smoke test.

This exists purely to validate the GPT architecture and train.py's training
loop (loss decreases, checkpoint save/reload round-trips) on a CPU in seconds,
before any real corpus (data/fetch_corpus.py) or Colab GPU time is involved.
Byte-level encoding means vocab_size=256, matching model.config.SMOKE_TEST_CONFIG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Small public-domain-style repeated text so the toy model has some
# learnable structure (repeated tokens/phrases) rather than pure noise.
_TOY_TEXT = """
The Lorentz factor gamma equals one over the square root of one minus beta squared.
Kinetic energy in special relativity equals gamma minus one times rest mass times c squared.
Energy equals mass times the speed of light squared.
The Schrodinger equation governs the time evolution of a quantum wavefunction.
Entropy of an isolated system never decreases in an isolated thermodynamic process.
Newton's second law states that force equals mass times acceleration.
The ideal gas law relates pressure, volume, temperature, and moles of gas.
Maxwell's equations describe the behavior of electric and magnetic fields.
""".strip() * 400  # repeat to give the tiny model enough tokens to train on


def main() -> None:
    p = argparse.ArgumentParser(description="Generate toy byte-level train.bin/val.bin for the smoke test")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "toy")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.frombuffer(_TOY_TEXT.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    train_data.tofile(args.out_dir / "train.bin")
    val_data.tofile(args.out_dir / "val.bin")

    print(f"Wrote {len(train_data):,} train tokens and {len(val_data):,} val tokens to {args.out_dir}")


if __name__ == "__main__":
    main()
