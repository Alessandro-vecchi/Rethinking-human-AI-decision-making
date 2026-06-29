# tests/

Mirror of `src/haidc/`. Tests are authored **per milestone, before** the implementation they
cover (`docs/conventions/TESTING.md`); a milestone is not done until its tests existed first and
pass. Run `make test`. Do not commit placeholder/`xfail` tests to clear a gate.

## `slow` marker (GPU-only backbone training)

A handful of M2 backbone tests **train the scratch resnet50** (`build_model` + `train_model`):
`test_backbone_overfit`, `test_backbone_determinism`, the training cases in
`test_backbone_val_select`, and `test_train_model_runs_through_device_path`. resnet50
forward+backward on CPU is prohibitive — a single 60-epoch overfit backward pass alone exceeds 30s
— so these are marked `@pytest.mark.slow` and **deselected by default** (`addopts = -m 'not slow'`
in `pyproject.toml`), with a 60s per-test `--timeout` guarding against any future hang. This is the
established "no CPU backbone tests" convention (`docs/HANDOFF.md` §3a): the resnet path is validated
on a Colab GPU, not on this box.

- `make test` / `pytest` → fast CPU suite (decision rule, cost accounting, metrics, split-hash,
  score/embedding contracts) — the behavior that gates the arms; green in seconds.
- `pytest -m slow` → the GPU backbone-training tests (run on Colab, where CUDA is available).
