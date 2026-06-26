# Command: run-milestone <M>

The orchestrator's per-milestone procedure. Keeps context lean and gates honest.

1. **Scope.** Read only `ROADMAP.md`'s row for `<M>` and `tasks/<M>-*.md`. Confirm its DAG
   predecessors are PASS. For M3/M4/M5, confirm the split is frozen+hash-verified and the backbone
   exists — else stop.
2. **Dispatch.** Spawn the owner agent (`ROADMAP.md`) with the ticket + the minimum doc pointers it
   names. For M3–M5, you may run up to three `arm-implementer` instances in parallel. Pass section
   numbers, not pasted docs.
3. **Gate.** When the subagent returns, spawn `verifier` on `<M>`. If REJECT, return the numbered
   failures to the owner; do not fix it yourself. Repeat until PASS.
4. **Record.** Append a `DECISIONS.md` entry (choices, SHAs, hyperparameters, `[U]→[V]`
   promotions). Update standing risks.
5. **Advance.** Re-read only the next milestone's gate criteria. Do not re-read the whole repo.

Stop-and-ask (surface to human): any `ROADMAP.md` standing risk fires, an invariant can't be held,
or a result would change the report's central claim.
