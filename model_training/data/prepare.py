"""Tokenizes data/raw/*.txt with the trained BPE tokenizer and packs the
result into train.bin/val.bin (uint16 token-id arrays) for train.py's memmap
data loader.

Run after fetch_corpus.py and tokenizer/train_tokenizer.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EOT_TOKEN = "<|endoftext|>"


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize and pack the raw corpus into train.bin/val.bin")
    p.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    p.add_argument("--tokenizer-path", type=Path, default=Path(__file__).parent.parent / "tokenizer" / "tokenizer.json")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--val-fraction", type=float, default=0.005)
    args = p.parse_args()

    if not args.tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found at {args.tokenizer_path} — run tokenizer/train_tokenizer.py first.")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    txt_files = sorted(args.raw_dir.rglob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found under {args.raw_dir} — run fetch_corpus.py first.")

    print(f"Tokenizing {len(txt_files)} files ...")
    all_ids: list[int] = []
    eot_id = tokenizer.token_to_id(EOT_TOKEN)

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        if eot_id is not None:
            all_ids.append(eot_id)
        print(f"  {txt_file.name}: {len(ids):,} tokens")

    arr = np.array(all_ids, dtype=np.uint16)
    split = int(len(arr) * (1 - args.val_fraction))
    train_ids, val_ids = arr[:split], arr[split:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(args.out_dir / "train.bin")
    val_ids.tofile(args.out_dir / "val.bin")

    print(f"Total tokens: {len(arr):,}  (train={len(train_ids):,}, val={len(val_ids):,})")
    print(f"Wrote train.bin/val.bin to {args.out_dir}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}  <- pass this to train.py's --vocab-size")


if __name__ == "__main__":
    main()
