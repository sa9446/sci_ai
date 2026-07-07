"""LocalTransformerRouter — drop-in replacement for NeuralSymbolicRouter that
runs the from-scratch model trained under model_training/ instead of calling
the Claude API.

Only useful once model_training/finetune.py has produced a Phase B
checkpoint (see model_training/README.md for the training roadmap — this is
expected to take weeks/months). Until then, sci_engine.config.settings
defaults to model_backend="anthropic" and this module is never imported.

Matches NeuralSymbolicRouter's public interface exactly — same
`async def route(query, *, retry_context=None) -> ScientificAnalysis` — so
main.py, InsulatedSandbox, and ResultGroundingValidator need zero changes to
work with either backend.

torch/tokenizers are imported lazily inside __init__ (not at module load
time) so importing sci_engine doesn't require them unless this backend is
actually selected.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from .agent import ScientificAnalysis
from .config import settings

logger = logging.getLogger(__name__)

_MODEL_TRAINING_DIR = Path(__file__).parent.parent / "model_training"


class LocalTransformerRouter:
    """Runs the self-trained GPT checkpoint instead of the Claude API."""

    def __init__(self) -> None:
        if not settings.local_model_checkpoint or not settings.local_model_tokenizer:
            raise RuntimeError(
                "model_backend='local' requires local_model_checkpoint and "
                "local_model_tokenizer to be set (see .env / Settings)."
            )

        if str(_MODEL_TRAINING_DIR) not in sys.path:
            sys.path.insert(0, str(_MODEL_TRAINING_DIR))

        import torch
        from tokenizers import Tokenizer

        from model import GPT, ModelConfig
        from schema_format import EOT_TAG, RESPONSE_FIELDS, RESPONSE_TAG, format_prompt

        self._torch = torch
        self._EOT_TAG = EOT_TAG
        self._RESPONSE_FIELDS = RESPONSE_FIELDS
        self._RESPONSE_TAG = RESPONSE_TAG
        self._format_prompt = format_prompt

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = Tokenizer.from_file(settings.local_model_tokenizer)

        ckpt = torch.load(settings.local_model_checkpoint, map_location=self._device)
        config = ModelConfig(**ckpt["config"])
        self._model = GPT(config).to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        logger.info(
            "LocalTransformerRouter loaded: %s params, device=%s, checkpoint=%s",
            f"{self._model.num_params():,}", self._device, settings.local_model_checkpoint,
        )

    # ── Public API — mirrors NeuralSymbolicRouter.route() exactly ────────────

    async def route(
        self,
        query: str,
        *,
        retry_context: str | None = None,
    ) -> ScientificAnalysis:
        return await asyncio.to_thread(self._route_sync, query, retry_context)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _route_sync(self, query: str, retry_context: str | None) -> ScientificAnalysis:
        full_query = query.strip()
        if retry_context:
            full_query += (
                f"\n\nThe previous attempt failed with this error, fix it:\n{retry_context}"
            )

        response_text = self._generate(full_query)

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LocalTransformerRouter produced invalid JSON: {exc}\nRaw output:\n{response_text[:500]}"
            ) from exc

        missing = [k for k in self._RESPONSE_FIELDS if k not in parsed]
        if missing:
            raise RuntimeError(f"LocalTransformerRouter output missing required keys: {missing}")

        return ScientificAnalysis(**{k: parsed[k] for k in self._RESPONSE_FIELDS})

    def _generate(self, query: str) -> str:
        torch = self._torch
        prompt_ids = self._tokenizer.encode(self._format_prompt(query)).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        eot_id = self._tokenizer.token_to_id(self._EOT_TAG)

        out_ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(settings.local_model_max_new_tokens):
                idx_cond = idx if idx.size(1) <= self._model.config.block_size else idx[:, -self._model.config.block_size:]
                logits, _ = self._model(idx_cond)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat((idx, next_id), dim=1)
                token_id = next_id.item()
                out_ids.append(token_id)
                if eot_id is not None and token_id == eot_id:
                    break

        full_text = self._tokenizer.decode(out_ids)
        return full_text.split(self._RESPONSE_TAG, 1)[-1].replace(self._EOT_TAG, "").strip()
