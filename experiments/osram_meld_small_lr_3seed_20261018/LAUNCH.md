# Fixed O6 configuration: actual three-seed extension

INTERNAL TEST-ORACLE SCREEN; per-rate ACC selection and same-forward UA/W-F1.

- Implementation commit: `465c5c7`, pushed to `YB123-DT/git_gcnet`,
  branch `feature/ecerc-features`.
- Server: `biggpu`; original cmerc-repro Python/Torch1.13.1+cu117,
  NumPy1.26.4/sklearn1.3.2; accepted original ECERC features.
- Reuse completed seed66 O6 without modifying or retraining it:
  ACC61.786398 / UA38.434973 / W-F1 58.917567.
- Only new runs: seeds67 and68, each50epochs, lr1e-5, Adam weight_decay2e-4.
- GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`, two parallel workers.
  GPU4 excluded; no other experiment was stopped, moved or modified.
- Persistent tmux: `meld_o6_3seed_1018`; controller PID2638023.
- Seed67 PID2638056; seed68 PID2638067.
- Immutable source:
  `/home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/source_465c5c7`.
- Output:
  `/home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1`.
- Controller log: experiment root `controller_v1.log`.
- Worker logs: `runs_v1/queue/logs/S67.attempt1.log` and `S68.attempt1.log`.

54 configuration/aggregation/optimizer regression tests and compilation passed.
Both workers passed immutable source, real environment, dataset hash and config
checks, bound the correct GPU UUID, constructed13,290,795-parameter original
Flat models and entered training. Initial state hashes differ, as expected for
independent seeds. This launch record does not claim completed50epoch results.

Both subsequently committed epoch1, including real forward/backward/optimizer
steps, eight rate selections and8 BEST checkpoint files per seed. All three
optimizer parameter groups were observed at LR1e-5 / weight_decay2e-4; JEPA loss
remained0. Queue states were running without failures. This is a training launch,
not merely a process-existence check.

Initialization hashes:

- Seed67: `ed27863d54e94fbb5a6df2a60205f37e154ecce246465a23dc42320c8a8a13a1`.
- Seed68: `a8b508e60385eef52b4d67f7a47f999cfef038d78423bf04f942d41f5d9cd79c`.

The controller automatically generates three-seed RESULTS.json/RESULT.md after
both new runs finish. Individual runs preserve8 BEST weights, exact selected
predictions/masks, class recall/support, mask traces and full last_training state.
Cross-seed mask equality is not required or claimed; the sampler algorithm is
unchanged. No HPO, added Block or more seeds follow this finite queue.

Run from the immutable source directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_small_lr_3seed_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1
```

Already running: do not repeat this command. Same-source recovery requires
explicit --resume and checking the existing controller/process identities first.
