# third_party/

Vendored upstream repos, used as the baseline implementations (do NOT re-implement — see docs/conventions/CODING.md):
- `okati2021/` — Okati, De & Gomez-Rodriguez 2021 (github.com/Networks-Learning/differentiable-learning-under-triage), SHA 43ec215
- `mozannar2020/` — Mozannar & Sontag 2020 (github.com/clinicalml/learn-to-defer), SHA e84f3ee

Pin each to a specific commit and record the SHA in docs/DECISIONS.md. Vendored code is read-mostly; wrappers live in src/haidc/arms/.
