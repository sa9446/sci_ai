"""Evaluation harness.

Phase A (--mode pretrain):  held-out perplexity, via the same get_batch/
    estimate_loss machinery train.py uses during training.

Phase B (--mode finetune):  generates a completion for each held-out query,
    parses it back into the ScientificAnalysis JSON schema, then runs it
    through the SAME InsulatedSandbox + ResultGroundingValidator main.py uses
    for Claude's output — so the reported schema-valid / sandbox-success /
    validator-pass rates are directly comparable to the Claude baseline
    before flipping sci_engine's model_backend setting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, ModelConfig
from schema_format import EOT_TAG, RESPONSE_TAG, RESPONSE_FIELDS, format_prompt
from train import estimate_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from sci_engine import InsulatedSandbox, ResultGroundingValidator  # noqa: E402


def eval_pretrain(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = ModelConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    losses = estimate_loss(model, args.data_dir, config.block_size, args.batch_size, args.device, args.eval_iters)
    for split, loss in losses.items():
        print(f"{split}: loss={loss:.4f}  perplexity={math.exp(loss):.2f}")


def _generate_completion(model: GPT, tokenizer: Tokenizer, query: str, device: str, max_new_tokens: int) -> str:
    prompt_ids = tokenizer.encode(format_prompt(query)).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eot_id = tokenizer.token_to_id(EOT_TAG)

    out_ids = prompt_ids[:]
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # greedy: deterministic for eval
        idx = torch.cat((idx, next_id), dim=1)
        token_id = next_id.item()
        out_ids.append(token_id)
        if eot_id is not None and token_id == eot_id:
            break

    full_text = tokenizer.decode(out_ids)
    return full_text.split(RESPONSE_TAG, 1)[-1].replace(EOT_TAG, "").strip()


def eval_finetune(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = ModelConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    sandbox = InsulatedSandbox()
    validator = ResultGroundingValidator()

    records = [json.loads(line) for line in args.eval_queries.read_text(encoding="utf-8").splitlines() if line.strip()]

    schema_valid = sandbox_success = validator_passed = 0
    total = len(records)

    for i, record in enumerate(records):
        query = record["query"]
        response_text = _generate_completion(model, tokenizer, query, args.device, args.max_new_tokens)

        try:
            parsed = json.loads(response_text)
            if not all(k in parsed for k in RESPONSE_FIELDS):
                raise ValueError("missing required keys")
            schema_valid += 1
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[{i}] schema-invalid: {exc}")
            continue

        exec_result = sandbox.execute(parsed["executable_python_code"])
        if not exec_result.success:
            print(f"[{i}] sandbox failed: {exec_result.error_type}: {exec_result.error_message}")
            continue
        sandbox_success += 1

        report = validator.validate(
            results=exec_result.result,
            domain=parsed["scientific_domain"],
            symbolic_repr=parsed["symbolic_representation"],
        )
        if report.passed:
            validator_passed += 1
        else:
            print(f"[{i}] validator failed: {[str(e) for e in report.errors]}")

    print()
    print(f"Total queries       : {total}")
    print(f"Schema-valid rate   : {schema_valid}/{total} ({100*schema_valid/max(1,total):.1f}%)")
    print(f"Sandbox-success rate: {sandbox_success}/{total} ({100*sandbox_success/max(1,total):.1f}%)")
    print(f"Validator-pass rate : {validator_passed}/{total} ({100*validator_passed/max(1,total):.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Phase A or Phase B checkpoint")
    p.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # pretrain mode
    p.add_argument("--data-dir", type=Path, help="[pretrain] dir with val.bin")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-iters", type=int, default=100)

    # finetune mode
    p.add_argument("--tokenizer-path", type=Path, default=Path(__file__).parent / "tokenizer" / "tokenizer.json")
    p.add_argument("--eval-queries", type=Path, help="[finetune] JSONL of held-out {\"query\": ...} records")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    if args.mode == "pretrain":
        if args.data_dir is None:
            p.error("--data-dir is required for --mode pretrain")
        eval_pretrain(args)
    else:
        if args.eval_queries is None:
            p.error("--eval-queries is required for --mode finetune")
        eval_finetune(args)


if __name__ == "__main__":
    main()
