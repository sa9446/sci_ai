"""Shared prompt/response formatting for Phase B fine-tuning and inference.

Both finetune.py (training) and sci_engine/local_router.py (inference, added
in Milestone 6) must agree byte-for-byte on this format, since the model
learns to reproduce it and local_router.py has to parse it back out.
"""
from __future__ import annotations

import json

QUERY_TAG = "<|query|>"
RESPONSE_TAG = "<|response|>"
EOT_TAG = "<|endoftext|>"

# Keys that make up the JSON the model must learn to emit — mirrors
# sci_engine.agent.ScientificAnalysis so local_router.py can construct one
# directly from the parsed dict.
RESPONSE_FIELDS = (
    "scientific_domain",
    "symbolic_representation",
    "variables",
    "expected_result_keys",
    "complexity_level",
    "executable_python_code",
)


def response_dict(record: dict) -> dict:
    return {k: record[k] for k in RESPONSE_FIELDS}


def format_prompt(query: str) -> str:
    return f"{QUERY_TAG}\n{query.strip()}\n{RESPONSE_TAG}\n"


def format_full_example(record: dict) -> str:
    """Full training sequence: prompt + target JSON + end-of-text marker."""
    prompt = format_prompt(record["query"])
    response = json.dumps(response_dict(record), indent=2)
    return f"{prompt}{response}\n{EOT_TAG}"


def split_prompt_len(record: dict, tokenizer) -> int:
    """Token length of just the prompt portion — used to mask the loss so
    the model isn't trained to predict the (fixed-format) query back."""
    return len(tokenizer.encode(format_prompt(record["query"])).ids)
