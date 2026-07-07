# sci_ai_engine model_training

A from-scratch GPT-style transformer, trained by hand (own architecture in
`model/gpt.py`, own training loop in `train.py`/`finetune.py`), intended to
eventually replace the Claude API call in `sci_engine/agent.py`. See
`C:\Users\Saman\.claude\plans\modular-seeking-wigderson.md` for the full plan
and rationale.

This is explicitly a **months-long project run on Google Colab**, not a
quick script. Progress is tracked here as milestones complete.

## Roadmap

- [x] Milestone 1 — Scaffold + local smoke test (architecture + training loop
      proven to work on a tiny toy corpus, CPU only)
- [ ] Milestone 2 — Real corpus (arXiv abstracts, Wikipedia STEM, OpenStax,
      filtered scientific code) + trained BPE tokenizer
- [ ] Milestone 3 — Phase A pretraining on Colab (GPT-2-124M-class model)
- [ ] Milestone 4 — Self-distillation dataset from the existing Claude-backed
      pipeline
- [ ] Milestone 5 — Phase B fine-tune on the distilled (query -> structured
      plan + code) dataset
- [ ] Milestone 6 — Eval harness + `local_router.py` integration into
      `sci_engine`
- [ ] Milestone 7 — Iterate (compare vs. Claude baseline, scale up if
      budget/time allow)

## Local setup

```
cd model_training
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-train.txt
```

## Milestone 1 smoke test (run locally, no Colab/GPU needed)

```
./.venv/Scripts/python.exe data/make_toy_data.py --out-dir data/toy
./.venv/Scripts/python.exe train.py --data-dir data/toy --out-dir checkpoints/smoke \
    --smoke-test --max-iters 300 --eval-interval 50 --save-interval 100 \
    --batch-size 8 --grad-accum-steps 1
```
Expect: `iter` loss printed every `--log-interval` steps trending down, an
`eval @ iter ...` line every `--eval-interval` steps, and
`checkpoints/smoke/ckpt_latest.pt` / `ckpt_best.pt` written to disk.

## Colab (Phase A / Phase B real runs)

Use `colab_notebook.ipynb`. It mounts Google Drive, installs
`requirements-train.txt`, and calls `train.py`/`finetune.py` with `--resume`
so re-running the notebook after a session disconnect just continues
training from the last saved checkpoint. **Always point `--out-dir` at a
Drive path**, never local Colab disk — local disk is wiped on disconnect.

## Cost / time log

(Fill in as Colab sessions run — helps gauge total budget/timeline against
the plan's "months" expectation.)

| Date | Milestone | Session length | Notes |
|------|-----------|-----------------|-------|
|      |           |                 |       |
