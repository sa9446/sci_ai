"""Phase B fine-tune: continues a Phase A checkpoint on the self-distilled
(query -> structured plan + code) dataset from distill_dataset.py, so the
from-scratch model learns to emit the same schema `ScientificAnalysis` uses.

Loss is masked on the prompt tokens (query + tags) — only the JSON response
tokens contribute to the gradient, standard SFT practice, and it means the
model isn't wasting capacity "learning" to predict fixed-format boilerplate
and user queries it will never need to generate itself.

Usage:
    python finetune.py --base-checkpoint checkpoints/phaseA/ckpt_best.pt \
        --dataset-path data/distilled.jsonl --tokenizer-path tokenizer/tokenizer.json \
        --out-dir checkpoints/phaseB --resume
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from model import GPT, ModelConfig
from schema_format import EOT_TAG, format_full_example, split_prompt_len
from train import lr_at_step, save_checkpoint, CHECKPOINT_NAME


class DistilledDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: Tokenizer, block_size: int) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        eot_id = tokenizer.token_to_id(EOT_TAG)
        dropped = 0

        for record in records:
            text = format_full_example(record)
            ids = tokenizer.encode(text).ids
            if eot_id is not None and (not ids or ids[-1] != eot_id):
                ids.append(eot_id)
            if len(ids) < 2:
                continue
            if len(ids) > block_size + 1:
                ids = ids[: block_size + 1]  # truncate over-long examples

            prompt_len = split_prompt_len(record, tokenizer)
            input_ids = ids[:-1]
            target_ids = ids[1:]
            # Mask the prompt portion of the loss (shifted by one, same as targets).
            target_ids = [
                (t if i >= prompt_len - 1 else -1) for i, t in enumerate(target_ids)
            ]

            # If truncation to block_size ate the entire response, every target
            # in this example is masked. A batch where ALL examples hit this
            # produces a 0/0 nan loss that permanently corrupts the optimizer's
            # momentum on backward() — so these examples must be dropped, not
            # silently kept.
            if all(t == -1 for t in target_ids):
                dropped += 1
                continue

            pad_len = block_size - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                target_ids = target_ids + [-1] * pad_len

            self.examples.append((input_ids, target_ids))

        if dropped:
            print(f"WARNING: dropped {dropped}/{len(records)} examples whose response "
                  f"was fully truncated by block_size={block_size} (query too long relative "
                  f"to block_size). Consider a larger --block-size or shorter queries.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Phase B fine-tune on the self-distilled dataset")
    p.add_argument("--base-checkpoint", type=Path, required=True, help="Phase A checkpoint to continue from")
    p.add_argument("--dataset-path", type=Path, required=True, help="JSONL from distill_dataset.py")
    p.add_argument("--tokenizer-path", type=Path, default=Path(__file__).parent / "tokenizer" / "tokenizer.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Resume from ckpt_latest.pt in --out-dir if present, else start from --base-checkpoint")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="Lower than Phase A pretraining LR, standard for continued fine-tuning")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    records = load_records(args.dataset_path)
    random.Random(args.seed).shuffle(records)
    split = int(len(records) * (1 - args.val_fraction))
    train_records, val_records = records[:split], records[split:]
    print(f"Loaded {len(records)} distilled examples ({len(train_records)} train / {len(val_records)} val)")

    latest_path = args.out_dir / CHECKPOINT_NAME
    start_ckpt = latest_path if (args.resume and latest_path.exists()) else args.base_checkpoint
    print(f"Loading weights from: {start_ckpt}")
    ckpt = torch.load(start_ckpt, map_location=device)
    config = ModelConfig(**ckpt["config"])

    train_ds = DistilledDataset(train_records, tokenizer, config.block_size)
    val_ds = DistilledDataset(val_records, tokenizer, config.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, betas=(0.9, 0.95))
    if start_ckpt == latest_path:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    start_iter = ckpt["iter_num"] + 1 if start_ckpt == latest_path else 0
    best_val_loss = ckpt["best_val_loss"] if start_ckpt == latest_path else float("inf")

    max_iters = args.epochs * len(train_loader)
    print(f"Model params: {model.num_params():,}  |  max_iters: {max_iters}  |  starting at {start_iter}")

    it = start_iter
    model.train()
    while it < max_iters:
        for x, y in train_loader:
            if it >= max_iters:
                break
            x, y = x.to(device), y.to(device)

            lr = lr_at_step(it, args.warmup_iters, max_iters, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if it % args.log_interval == 0:
                print(f"iter {it:6d}/{max_iters} | loss {loss.item():.4f} | lr {lr:.2e}")

            if it > 0 and it % args.save_interval == 0:
                val_loss = _eval(model, val_loader, device) if len(val_ds) else None
                if val_loss is not None:
                    print(f"  eval @ iter {it}: val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(args.out_dir / "ckpt_best.pt", model, optimizer, config, it, best_val_loss)
                        print(f"  new best val loss {best_val_loss:.4f} -> saved ckpt_best.pt")
                save_checkpoint(args.out_dir / CHECKPOINT_NAME, model, optimizer, config, it, best_val_loss)
                model.train()

            it += 1

    save_checkpoint(args.out_dir / CHECKPOINT_NAME, model, optimizer, config, max_iters - 1, best_val_loss)
    print(f"Fine-tuning complete. Final checkpoint saved to {args.out_dir / CHECKPOINT_NAME}")


@torch.no_grad()
def _eval(model: GPT, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
    return total / max(1, count)


if __name__ == "__main__":
    main()
