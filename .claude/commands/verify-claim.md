# Command: verify-claim

Use before stating any method/dataset fact that is not already `[V]` in `GROUND_TRUTH.md`.

1. Locate the claim's source. The HCT papers (2602.02375, 2603.29866) and Okati (2103.08902) are in
   the project as image-scan archives; Mozannar (2006.01862) + repos are on arXiv/GitHub.
2. Open the actual page/section. Read it. Do not rely on recall for a number, an equation, or a
   section pointer.
3. If confirmed: state it, cite the exact location (`CITATIONS.md`), and if it was a `[U]` in
   `GROUND_TRUTH.md`, promote it to `[V]` with the citation in the same change.
4. If you cannot confirm it: keep it `[U]`, state the claim without a fabricated pointer, and flag
   it for a human. Never approximate a citation to sound confident.

A fabricated citation is a hard failure (`CITATIONS.md`). When unsure, the correct output is "I
could not verify this", not a plausible-looking reference.
