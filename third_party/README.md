# third_party/

Vendored upstream repos, used as the baseline implementations (do NOT re-implement — see docs/conventions/CODING.md):
- `differentiable-learning-under-triage/` — Okati, De & Gomez-Rodriguez 2021 (github.com/Networks-Learning/differentiable-learning-under-triage)
- `learn-to-defer/` — Mozannar & Sontag 2020 (github.com/clinicalml/learn-to-defer)

Pin each to a specific commit and record the SHA in docs/DECISIONS.md. Vendored code is read-mostly; wrappers live in src/haidc/arms/.
