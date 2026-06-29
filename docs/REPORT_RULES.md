# REPORT_RULES.md — Writing the Report

> Style and structure contract for the technical report. Read this (and
> `docs/GROUND_TRUTH.md`, plus `HANDOFF.md` and `DECISIONS.md`, for ground-truth
> facts) before editing a single sentence of the report. The report already
> exists: `docs/REPORT.tex` is the production LaTeX, `docs/REPORT.md` is its
> Markdown source (kept verbatim-equivalent), and `docs/REPORT.pdf` is compiled.

---

## 1. Hard constraints

- **2–3 pages.** (`ROADMAP.md` M7; the `REPORT.tex` header.) Ruthlessly cut. If
  it doesn't earn its space, remove it.
- **Fixed structure — do not invent or reorder sections.** This is the structure
  already in `docs/REPORT.tex`:
  1. **Framing and contribution** — the head-to-head novelty (HCT vs L2D on a
     shared dataset, backbone, and split), with the **null result stated up front**.
  2. **Dataset and the non-expert-rater caveat.**
  3. **Methods and the invariants that make the comparison fair** — shared backbone,
     frozen split, the logged Option-B deviation.
  4. **Results.**
  5. **Central finding (an honest null).**
  6. **Limitations.**
- Honest framing of scope is part of the report, not an aside. State plainly, in
  both Results and Limitations: accuracy here is **agreement with crowd consensus,
  not truth**; the raters are **non-expert citizen-science volunteers**; the
  content-based crosswalk reaches only **59% coverage** (4,621 of 10,000); and the
  whole study is **a single dataset on a single frozen backbone**. These are
  disclosed limitations, not footnotes.

## 2. Anti-hallucination (non-negotiable)

- **Zero invented facts.** No fabricated metrics, numbers, citations, or
  results. Every methodological claim and every number must trace to the code
  or a context file (`docs/GROUND_TRUTH.md` and family).
- **Placeholder protocol.** When you lack a specific value, insert a visible
  tag — never a plausible guess:
  - `[INSERT METRIC: e.g. AI-alone test accuracy with 95% bootstrap CI]`
  - `[CITATION NEEDED: HCT cost-savings claim, arXiv:2602.02375]`
  - `[INSERT FIGURE: Pareto frontier — accuracy vs expected human-query cost]`
  At the end of each draft, append a **Missing Context Checklist** listing
  every placeholder the user must fill.
- **Halt and ask.** If you cannot complete a paragraph from available context,
  stop, name the missing information precisely, and ask. Do not write filler.

## 3. Anti-LLM style guide

Academic, rigorous, and recognisably human. Banned AI tropes:

- **Banned vocabulary:** delve, leverage, nuanced, fostering, underscore,
  multifaceted, tapestry, realm, pivotal, seamlessly, paradigm, testament.
  (Use only if a precise technical definition genuinely requires it.)
- **No hollow intensifiers.** "significant results", not "highly significant
  results"; drop "crucially", "importantly", "notably" when they add nothing.
- **No faux profundity.** No sweeping openers or philosophical conclusions
  about the nature of human–AI collaboration or trust. Stay on the concrete work.
- **No synonym stacking.** "capabilities", not "skills and capabilities";
  "tested", not "tested and evaluated".
- **No forced rule-of-three.** Group items in threes only if the data does.
- **Avoid `—` and `-` as connectors.** Prefer rephrasing into clean
  sentences; use them only when genuinely unavoidable.

## 4. Humanized prose mechanics

- **Vary sentence length.** Mix short, direct sentences with longer
  compound-complex ones. Do not default to uniform medium-length sentences.
- **Active voice by default.** "Every arm consumed the same frozen backbone
  scores", not "The same scores were consumed by every arm".
- **Concrete subjects and verbs.** Avoid nominalisation: "we measured X", not
  "measurement of X was conducted".
- **Prose over lists.** Default to paragraphs. Use a list only for a true
  sequence of steps, a formal set of constraints, or when explicitly asked.
- **Minimal formatting.** Sparse bold/italics; let sentence structure carry
  emphasis.

## 5. Workflow (drafter discipline)

1. Read the relevant context (`docs/GROUND_TRUTH.md`, `docs/HANDOFF.md`,
   `docs/DECISIONS.md`, `docs/conventions/CITATIONS.md`). Do not write from memory.
2. The report already exists, so most work is **editing or extending** it, not
   drafting from blank. For any new section, produce a **bulleted outline /
   skeleton** first — driving focus of each paragraph plus the transition into
   the next. **No full prose yet.**
3. **Pause.** Ask: "Does this structure work? Shall I draft this section?"
4. Draft only the approved section, using the Placeholder Protocol.
5. Treat `docs/REPORT.tex` and `docs/REPORT.md` as the **protected production
   report** — edit them only on explicit instruction, and keep the two in sync
   (the `.tex` header states it is a verbatim conversion of the `.md`). Keep
   unapproved drafts out of those files (write to a `drafts/` location or show
   inline). Citations follow the **arXiv-identifier** convention (`\arxiv{...}`,
   per `docs/conventions/CITATIONS.md`); there is no `.bib`.

## 6. Map text to the real work

When writing Framing/Results, anchor claims to actual repo artifacts: the five
arms and baselines in `docs/REPORT.tex` and `results/` — **AI-alone**,
**single-human**, **HCT**, **L2D-Okati** (learned, plus the non-deployable
**oracle**), and **L2D-Mozannar** — over one shared, frozen ResNet-50 backbone.
Every number traces to `results/tables/summary.csv` or
`results/figures/pareto_frontier.png`. The comparison is held fair by the
**shared-backbone + frozen-split invariants** (`GROUND_TRUTH.md` §7) and the
logged **Option-B deviation** (freeze the backbone, learn only the rejector;
`docs/DECISIONS.md`).

Be candid about gaps, but characterize them precisely:
- **De Toni cost–accuracy framing** — adopted as **framing only, not implemented**.
  Say so; do not present it as an evaluated arm.
- **Opinion-leader (ρ) reconstruction** — **not supported** by aggregate vote
  counts. Because $h_1 \perp h_2 \mid x$, `h2` is a second *draw*, not a second
  *human* (`HANDOFF.md` §6). Frame this as a principled scoping decision, not a failure.
- **AI-as-advisor baseline** — **out of scope** for the one run unless the vendored
  L2D code supports it cleanly (`ROADMAP.md` baselines note).
- **Option B** — triage-only over a frozen backbone: it removes a backbone confound
  but does **not** evaluate the papers' end-to-end co-training, and their co-trained
  accuracy claims are not imported here.
Honesty about gaps is part of the Limitations section, not something to paper over;
every number in the report traces to an artifact under `results/`.
