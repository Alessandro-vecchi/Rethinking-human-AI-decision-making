# conventions/CITATIONS.md

Inventing a citation, a quote, a section number, or an equation reference is a **hard failure**.
A wrong citation in a research report is worse than an admitted gap.

## Rules
- Cite the **specific** location: section, equation, theorem, table, or footnote — not just the
  paper. Good: "Okati 2021, Thm. 3"; "Okati 2021, fn. 8"; "HCT-1, abstract". Bad: "Okati shows…".
- Every method/dataset claim in code comments, docs, or the report must trace to
  `GROUND_TRUTH.md` (which is itself cited) or to a source you have opened and checked.
- If you cannot verify a section/equation pointer, write the claim **without** a fabricated pointer
  and mark it `[U]` for a human to confirm. Do not approximate a citation to look authoritative.
- Quoting: paraphrase by default. If an exact phrase is load-bearing (e.g. a defined term), keep
  any quote short and attribute it precisely. Do not reproduce figures, tables, or long passages.
- Distinguish what a paper *claims about itself* from what *this project measured*. HCT's "28–44%
  cost saving" is HCT-1's claim on its own datasets, **not** a result on Galaxy Zoo. Never present
  an upstream claim as a finding of this study.

## Canonical references (use these identifiers)
- HCT-1: Berger et al., arXiv:2602.02375, 2026.
- HCT-2: Berger et al., arXiv:2603.29866, 2026.
- Okati: Okati, De & Gómez-Rodríguez, NeurIPS 2021, arXiv:2103.08902.
- Mozannar: Mozannar & Sontag, ICML 2020, arXiv:2006.01862.
- De Toni et al., NeurIPS 2024 (framing only; not an arm).

When in doubt about a pointer, open the file in the project / the arXiv page and check. The two HCT
papers and Okati are in the project as image scans; read the page rather than recalling it.
