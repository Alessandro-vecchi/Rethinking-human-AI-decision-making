# Single-command reproduction. Each target is deterministic given configs/ + seed.
# Targets must fail loudly on non-determinism (see docs/conventions/REPRODUCIBILITY.md).
.PHONY: help env test lint data backbone arms eval report all clean verify-split

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-14s %s\n", $$1, $$2}'

env:           # create venv + install pinned deps
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:          # run unit tests (must pass before any arm is marked done)
	pytest -q --cov=src/haidc

lint:          # static checks
	ruff check src tests

data:          # build frozen Galaxy Zoo split + multi-rater label table (M1)
	python -m haidc.data.build_split --config configs/data.yaml

verify-split:  # assert the split manifest matches the recorded hash (guards confounds)
	python -m haidc.data.verify_split --config configs/data.yaml

backbone:      # train/load the SHARED AI classifier reused by every arm (M2)
	python -m haidc.arms.backbone --config configs/backbone.yaml

arms:          # run HCT, L2D-Okati, L2D-Mozannar, baselines (M3-M5)
	python -m haidc.arms.run_all --config configs/arms.yaml

eval:          # compute metrics + Pareto frontier + plots (M6)
	python -m haidc.eval.pareto --config configs/eval.yaml

report:        # assemble the 2-3 page technical report from results/ (M7)
	@echo "See tasks/M7-report.md"

all: data verify-split backbone arms eval report
clean:
	rm -rf results/figures/* results/tables/* .pytest_cache .ruff_cache
