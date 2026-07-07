"""Trains a byte-level BPE tokenizer on the Phase A corpus.

Run after data/fetch_corpus.py has produced raw text files. Output
(tokenizer.json) is consumed by data/prepare.py to pack the corpus into
token-id shards, and its vocab_size feeds train.py's --vocab-size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema_format import EOT_TAG, QUERY_TAG, RESPONSE_TAG  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on raw corpus text files")
    p.add_argument("--corpus-dir", type=Path, default=Path(__file__).parent.parent / "data" / "raw",
                    help="Directory of .txt files produced by fetch_corpus.py")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--vocab-size", type=int, default=32_000,
                    help="Smaller than GPT-2's 50257 since the corpus is domain-narrow (science/math/code)")
    p.add_argument("--min-frequency", type=int, default=2)
    args = p.parse_args()

    files = sorted(str(f) for f in args.corpus_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(
            f"No .txt files found under {args.corpus_dir} — run data/fetch_corpus.py first."
        )
    print(f"Training BPE tokenizer on {len(files)} files ...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        # EOT_TAG/QUERY_TAG/RESPONSE_TAG must be atomic tokens, not split into
        # byte-level sub-tokens, since schema_format.py's split_prompt_len()
        # counts tokens up to RESPONSE_TAG to mask the loss on the prompt —
        # imported from schema_format so this can't drift out of sync with it.
        special_tokens=[EOT_TAG, "<|pad|>", QUERY_TAG, RESPONSE_TAG],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out_dir / "tokenizer.json"))
    print(f"Saved tokenizer.json (vocab_size={tokenizer.get_vocab_size()}) to {args.out_dir}")


if __name__ == "__main__":
    main()
