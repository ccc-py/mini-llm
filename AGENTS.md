# mini-llm — AGENTS.md

## Repo structure

Toy LLM pretrain+finetune+distill in pure PyTorch. Three independent stages, each self-contained (run from within its directory):

| Directory | What it does |
|-----------|-------------|
| `v1-pretrain/` | Single-stage character LM: `python mini-llm.py --file input.txt` |
| `v2-finetune/` | Two-stage: data gen → pretrain → finetune. Model in `model.py` |
| `v3-distill/` | Same as v2 but data generated via NVIDIA API distillation |
| `tool/` | (Separate) HuggingFace corpus builder (`datasets` needed) |

## Commands (run from the subdirectory)

v1: `python mini-llm.py --file input.txt` (optional: `--iters N --seq_len N --batch_size N --gen_len N`)

v2: `./run.sh` → picks one data generator (edit run.sh to swap), then runs pretrain.py → finetune.py

v3: same as v2, but `gen_data_distill.py` requires `NVIDIA_API_KEY` env var

## Workflow rules

- **Order matters** (within each stage): data generation → `pretrain.py` → `finetune.py`
- **Finetune depends on pretrain**: `finetune.py` loads `pretrain.pt` weights
- **Data generators are mutually exclusive**: `gen_data_wuxia.py`, `gen_data_rule.py`, `gen_data_robot.py`, `gen_data_distill.py` — only one should run per session; they all write the same output files (`vocab.pkl`, `pretrain_data.pt`, `finetune_data.pt`)
- **Character-level tokenizer**: built from unique chars in the corpus each time you generate data; `vocab.pkl` is a dict with `stoi`, `itos`, `vocab_size`

## Artifacts

All auto-generated and gitignored: `*.pt`, `*.pkl`, `pretrain.txt`, `finetune.txt`

Generated from data scripts and overwritten on each run.

## Model

Architecture: RoPE, RMSNorm, SwiGLU FFN, weight tying. Defaults: d_model=128, n_heads=4, seq_len=64.
- v1: 3 layers, ~0.8M params, trained 2000 steps
- v2/v3: 4 layers, trained 500 steps pretrain + 300 steps finetune

Runs on CPU by default (falls back from CUDA).

## No dependencies

No requirements.txt, no pyproject.toml. Only `torch` needed. `tool/build_corpus.py` additionally needs `datasets` from HuggingFace.

## Testing, linting, typechecking

None.
