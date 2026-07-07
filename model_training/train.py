"""Phase A pretraining loop for the from-scratch GPT.

Designed for the Colab reality of this project: sessions disconnect (12h cap,
idle timeouts) and this is a multi-week/month schedule, so checkpointing to
disk (point --out-dir at a mounted Google Drive path) and --resume are not
optional extras — every run should be assumed to end unexpectedly and be
restarted with the same command plus --resume.

Usage (local smoke test, Milestone 1):
    python data/make_toy_data.py --out-dir data/toy
    python train.py --data-dir data/toy --out-dir checkpoints/smoke --smoke-test \
        --max-iters 300 --eval-interval 50 --save-interval 100

Usage (Colab, Phase A real run):
    python train.py --data-dir /content/drive/MyDrive/sci_ai_engine/data \
        --out-dir /content/drive/MyDrive/sci_ai_engine/checkpoints/phaseA \
        --resume
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, ModelConfig, SMOKE_TEST_CONFIG

CHECKPOINT_NAME = "ckpt_latest.pt"
BEST_CHECKPOINT_NAME = "ckpt_best.pt"


def get_batch(data_dir: Path, split: str, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a random batch from a packed uint16 .bin shard via memmap.

    Re-opening the memmap on every call (rather than caching it) avoids a
    known memory-leak footgun where a long-lived np.memmap slowly grows the
    process's resident set over a multi-day training run.
    """
    path = data_dir / f"{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")

    if len(data) <= block_size + 1:
        raise ValueError(
            f"{path} has only {len(data)} tokens, which is <= block_size+1 "
            f"({block_size + 1}). Use a smaller --block-size or a larger corpus."
        )

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: Path, block_size: int, batch_size: int, device: str, eval_iters: int) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        if not (data_dir / f"{split}.bin").exists():
            continue
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data_dir, split, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at_step(step: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_checkpoint(out_dir: Path, device: str) -> dict | None:
    ckpt_path = out_dir / CHECKPOINT_NAME
    if not ckpt_path.exists():
        return None
    print(f"Resuming from checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, config: ModelConfig, iter_num: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — avoids a truncated/corrupt checkpoint
    # if a Colab disconnect happens mid-write.
    tmp_path = path.with_suffix(".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase A pretraining for the from-scratch sci_ai_engine GPT")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing train.bin / val.bin")
    p.add_argument("--out-dir", type=Path, required=True, help="Checkpoint directory (point at Drive for real runs)")
    p.add_argument("--resume", action="store_true", help="Resume from ckpt_latest.pt in --out-dir if present")
    p.add_argument("--smoke-test", action="store_true", help="Use the tiny SMOKE_TEST_CONFIG instead of full-size model")

    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--grad-accum-steps", type=int, default=8, help="Simulates a larger batch size on small Colab GPUs")
    p.add_argument("--block-size", type=int, default=None, help="Overrides config default if set")

    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--vocab-size", type=int, default=None, help="Set from tokenizer.get_vocab_size(); required unless --smoke-test")
    p.add_argument("--n-layer", type=int, default=None, help="Overrides ModelConfig default (12); useful for quick local iteration before the full Colab run")
    p.add_argument("--n-head", type=int, default=None, help="Overrides ModelConfig default (12)")
    p.add_argument("--n-embd", type=int, default=None, help="Overrides ModelConfig default (768)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    print(f"device={device}  dtype={dtype}")

    if args.smoke_test:
        config = SMOKE_TEST_CONFIG
    else:
        if args.vocab_size is None:
            p.error("--vocab-size is required unless --smoke-test is set (use tokenizer.get_vocab_size())")
        config = ModelConfig(vocab_size=args.vocab_size)
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    if args.n_head is not None:
        config.n_head = args.n_head
    if args.n_embd is not None:
        config.n_embd = args.n_embd
    config.__post_init__()  # re-validate n_embd % n_head after any CLI overrides

    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, betas=(0.9, 0.95))

    start_iter = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.out_dir, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_iter = ckpt["iter_num"] + 1
            best_val_loss = ckpt["best_val_loss"]
        else:
            print("--resume set but no checkpoint found; starting from scratch")

    print(f"Model params: {model.num_params():,}")
    print(f"Starting at iter {start_iter}, best_val_loss so far: {best_val_loss:.4f}")

    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = lr_at_step(it, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(args.data_dir, "train", config.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {it:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt*1000/max(1,args.log_interval):.1f}ms/iter")
            t0 = time.time()

        if it > 0 and it % args.eval_interval == 0:
            losses = estimate_loss(model, args.data_dir, config.block_size, args.batch_size, device, args.eval_iters)
            print(f"  eval @ iter {it}: {losses}")
            if "val" in losses and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(args.out_dir / BEST_CHECKPOINT_NAME, model, optimizer, config, it, best_val_loss)
                print(f"  new best val loss {best_val_loss:.4f} -> saved {BEST_CHECKPOINT_NAME}")

        if it > 0 and it % args.save_interval == 0:
            save_checkpoint(args.out_dir / CHECKPOINT_NAME, model, optimizer, config, it, best_val_loss)
            print(f"  saved checkpoint at iter {it} -> {CHECKPOINT_NAME}")

    # Final checkpoint on normal completion.
    save_checkpoint(args.out_dir / CHECKPOINT_NAME, model, optimizer, config, args.max_iters - 1, best_val_loss)
    print(f"Training complete. Final checkpoint saved to {args.out_dir / CHECKPOINT_NAME}")


if __name__ == "__main__":
    main()
