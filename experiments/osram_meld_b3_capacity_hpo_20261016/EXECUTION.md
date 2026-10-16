# Corrected scope: four other-knob trials actually launched

INTERNAL TEST-ORACLE SCREEN; no final performance results yet.

The previous GPU3 extension still varied only tokens and rounds. The user
corrected that interpretation. It finished normally without being killed; its
results remain separate. The following four new trials genuinely vary B3
dimension, internal attention heads, FFN width and adapter-specific learning rate.
These are joint conditional HPO proposals, not single-factor ablations.

Numerical/controller source: `6ce8ca60d507fe81dd37d9ce74623104ee2e7098`.
Immutable checkout:
`/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_code_6ce8ca6`.
Output:
`/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_hpo_20261016`.
Server: `ssh biggpu`; tmux: `meld_b3_capacity_gpu3_1016`; controller PID1854477.
Physical GPU3 UUID: `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`.

| Job | Tokens | Rounds | B3 dim | B3 heads | FFN hidden | B3 LR | Worker PID |
|---|---:|---:|---:|---:|---:|---:|---:|
| b3_capacity_01 | 4 | 3 | 64 | 2 | 512 | 0.0001 | 1858170 |
| b3_capacity_02 | 2 | 3 | 256 | 8 | 128 | 0.0003 | 1858182 |
| b3_capacity_03 | 3 | 3 | 64 | 2 | 128 | 0.0003 | 1858206 |
| b3_capacity_04 | 5 | 4 | 256 | 8 | 512 | 0.0001 | 1858223 |

All four queue records were verified concurrently running. The required
nondefault axes for jobs1–4 are respectively dim,heads,FFN,LR; other fields are
also TPE-selected. Full configurations, source hashes, reserved previous trials
and SQLite state are retained remotely. Only four new jobs are authorized here,
bringing the separately tracked total to28, not an unlimited search.

## Unchanged settings

MELD seed66,50epochs,batch32,global lr0.001,WD1e-5,dropout0.5,original Flat1600,
raw Local Skip,CHM8x64 read/write and queries,task head/loss,cyclic0.0–0.7 and
`gcnet-random-mask-v1`. The smaller B3 LR applies to the entire existing
`osram.meaningful_block`,including tokenizer/core/raw decoders,never Flat or CHM.
Per-rate Test-best ACC retains matching selected predictions/masks and UA.
The paired-mask objective remains separate and unchanged.

## Verification

- Local combined suite:43 passed,1 skipped for unavailable local Optuna.
- Deployed remote suite:26 passed,none skipped,including actual SQLite/TPE,
  full-tuple uniqueness,non-token-only coverage,finite budget and resume tests.
- Historical default B3 state/RNG/nonzero-decoder outputs and strict checkpoint
  loading are unchanged. Tests cover nondefault capacities,inactive Gap,padding,
  finite gradients,unchanged raw Local Skip and optimizer-group isolation.
- New largest structure real MELD batch smoke:tokens8/rounds4/dim256/heads8/
  FFN512,three backward/optimizer steps; complete. Peak3204.083MiB,15.473seconds;
  nonzero B3 gradient tensor counts0/18/343 across the steps. No performance
  conclusion is drawn from this check.
- Existing optional torch-sparse shared-library warning remains nonfatal and
  unrelated to this model path. Syntax/whitespace checks passed.

The old24 outcomes provide observations only at the historical defaults for the
four newly exposed knobs. They do not establish which new capacity or LR is best;
these four new trials are a small exploratory batch,not a fully optimized result.
