# Single-command reproduction. Each target is deterministic given configs/ + seed.
# Targets must fail loudly on non-determinism (see docs/conventions/REPRODUCIBILITY.md).
.PHONY: help env test lint data backbone arms eval report all clean verify-split

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-14s %s\n", $$1, $$2}'

env:           # create venv + install pinned deps + editable package
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e . --no-deps

test:          # run unit tests (must pass before any arm is marked done)
	pytest -q --cov=src/haidc

lint:          # static checks
	ruff check src tests

data:          # build frozen Galaxy Zoo split + multi-rater label table (M1)
	python -m haidc.data.build_split --config configs/data.yaml

verify-split:  # assert the split manifest matches the recorded hash (guards confounds)
	python -m haidc.data.verify_split --config configs/data.yaml

backbone:      # GPU/Colab ONLY: train shared classifier + export scores & embeddings (M2/Stage-B)
	# Produces results/backbone_scores.parquet + results/backbone_embeddings.parquet — the frozen
	# interface every arm consumes. Requires a GPU + the raw images; ran on Colab, outputs committed.
	# NOT part of `make all` (see REPRODUCE.md). Do not run on this CPU box (tests are marked `slow`).
	python -m haidc.arms.backbone --config configs/backbone.yaml

arms:          # run HCT, L2D-Okati, L2D-Mozannar, baselines (M3-M5)
	python -m haidc.arms.run_all --config configs/arms.yaml

eval:          # compute metrics + Pareto frontier + plots (M6)
	python -m haidc.eval.pareto --config configs/eval.yaml

report:        # assemble the 2-3 page technical report from results/ (M7)
	@echo "See tasks/M7-report.md"

# `all` reproduces the report's figure + table on a clean CPU checkout from the COMMITTED frozen
# artifacts (data/label_table.parquet + results/backbone_{scores,embeddings}.parquet): it re-runs
# the arms (M3-M5) then eval (M6), deterministically (two runs identical). The preprocessing that
# PRODUCED those artifacts — `data`/`verify-split` (rebuild the label table from the raw Willett/
# Kaggle catalogs in data/raw, ~161M, NOT committed) and `backbone` (GPU/Colab) — is upstream of the
# frozen interface and is documented in docs/REPRODUCE.md, not run here.
all: arms eval report
clean:          # remove regenerable figures/tables/caches; NEVER the committed frozen *.parquet
	rm -rf results/figures/* results/tables/* .pytest_cache .ruff_cache
