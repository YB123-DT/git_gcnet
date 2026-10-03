# Twenty external Local/Base/Gap candidates: implementation plan

INTERNAL DIAGNOSTIC ONLY

WITHDRAWN BEFORE TRAINING. User rejected the basic-operator shortlist and requested
substantial complete blocks. The following is historical, not an active launch
instruction. No deployment/training occurred; code remains default-off archival work.

The user authorized source screening, implementation, and one seeded training run
per method. Keep revision5eb6061's cfg84 Flat as the intact anchor. Add an optional
`osram_readout_candidate=none` switch and an isolated readout residual, not a new
Memory trajectory or loss. These are explicitly adapted mechanisms, not full
reproductions of the source papers.

Use canonical seed66 MOSI baseline configuration:100 epochs, batch32, Adam .001,
weight decay .00001, original task MSE, cyclic random missing0–.7, same evaluation
masks and per-rate BEST Test-oracle. Report selection bias from screening twenty
methods; no formal generalization or mechanism claim follows from the winner.

Common interface: projected128-dimensional tokens `[Local,Base,GapA,GapT,GapV]`,
forward half only; safe input/output masking, inactive Gap only when unavailable,
padding and first valid utterance residual exactly zero. Each distinct mechanism
returns128 features; zero-initialized Linear128→actual output dimension adds to the
original Flat pre-norm anchor. No additional stochastic dropout. Isolate module
initialization RNG. Keep original Memory formulas, queries, task head and losses;
no existing gates, Relation, completion, JEPA, paired views or frozen parameters.
Parameter budgets differ and must be reported.

- [ ] Verify twenty distinct papers and author-code formulas in three JSON files;
      every entry has URL, author array, source-code evidence and adaptation limits.
- [ ] Write failing unit tests, implement independently scoped candidate modules,
      and test masking/NaNs/first history/gradients/updates/zero initialization.
- [ ] Wire default-off config/model/OSRAM/checkpoint construction; test original
      initialization, RNG, predictions, strict legacy loading and one causal scan.
- [ ] Test immutable single-run configuration and bounded durable queue behavior.
- [ ] Lock manifest, record parameter counts, run correctness checks and commit/push.
- [ ] Snapshot source on biggpu; smoke then scale to at most five concurrent jobs
      on healthy GPU5 after checking actual memory/throughput. Never use GPU4.
      GPUs0–3 currently have unrelated queued work;6–7 are compute-saturated.
- [ ] Retain config, hashes, environment, PID/UUID, predictions,100-epoch history,
      eight BEST checkpoints and failures. Do not overwrite outputs or rerun a
      completed configuration. No protocol changes to fit concurrency.
- [ ] Report all twenty outcomes: per-rate W-F1, eight-rate/high-missing means and
      paired seed66 Flat deltas. No performance-driven hyperparameter expansion.

Owned paths: new `gcnet_missing_m3/readout_candidates*.py`, focused integration in
existing model/OSRAM/trainer/checkpoint builder, `tests/test_readout_candidate*.py`,
and this experiment directory. Preserve unrelated dirty `gcnet/model.py`.
Use existing local multimodalerc310 for tests and remote s0 for training; no new
dependencies, local training, downloaded checkpoint copies or Git weight uploads.
