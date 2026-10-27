# CLAUDE.md

Read the full project context before doing anything:

@context.md

## Working rules for Claude Code in this repo

- Work through the experiments in order (E0 → E1 → E2 → E3). Do not start the next one until the current one's PASS/GO check is written up in `context.md` → "Status and decisions log".
- Before writing new code, read the existing file you are changing. `MapAnything_Exp1/exp1/model_io.py` is the only place that may call the MapAnything API.
- Every new corruption type needs: a config entry, a manifest entry, and a unit test in `MapAnything_Exp1/tests/`.
- Run on the GPU server only. Never modify shared conda environments. Use the `mapanything` env.
- Never change `configs/pilot.json` (or any eval grid) after eval-split results exist. Debug on `dev__*` clips only.
- Never write a number in a report that did not come from a results file. Link the file.
- When unsure about MapAnything internals, read the installed source and cite file:line. Do not guess.
