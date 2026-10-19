# MOSI GCNet-missing Flat versus one-stage Text-source Router

INTERNAL EXPLORATORY EXPERIMENT — no performance claim before completion.

User authorization: six fresh runs, seeds66/67/68 × Flat/Router, 150 epochs.
The only intended data-protocol change from the pending MOSI experiment is
`conversation-mask-v2` → `gcnet-random-mask-v1`. The old 80.747 three-seed
per-rate Test-best score is historical context, not a matched baseline.

## Fixed method and optimization

- Current cfg84 no-JEPA Large Flat1600, Local Skip, original causal CHM and
  joint observed-modality Block Delta Write, write_step0.6, 8×64 heads.
- Current one-stage Text-source Router: one scan, full/Base-retained/
  Gap-retained scalar candidates; only eligible current-T-missing and
  strictly-past-T-observed rows are mixed. No new Gate/JEPA/completion/loss.
- MOSI original scalar MSE. Both model families train from scratch.
- Adam, batch32, weight_decay1e-5, original dropout; backbone LR1e-3 for
  epochs1–100 and1e-4 for101–150. Router LR3e-4 then3e-5.
- Training seeds vary model initialization, data ordering and train RNG.
  Split and fixed evaluation banks use seed66 across the three seeds.
- Historical model factory uses fold5 initialization namespace; actual MOSI
  data fold is1. These are recorded separately, not silently conflated.
- GCNet host/guest masks are separately drawn on each padded batch rectangle;
  effective availability uses original speaker selection. At rate0.7 exactly
  one modality remains per valid utterance; padding is zero. Cyclic0.0–0.7.
- One checkpoint per run selected by mean eight-rate validation W-F1,
  tie-breaking by lower MSE; test only evaluates that selected checkpoint.
  This preserves the just-completed Router experiment's selection protocol,
  not old per-rate Test-best selection.

## Runtime and provenance

Server: biggpu. Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`;
Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2. No server/environment migration.
`num_workers=0`; pre-extracted feature Dataset resides in CPU RAM, not videos.

Accepted original MOSI DATA.json SHA256:
`e790eee2c850418fde9652ab85ca4a1eeb12b155c5651d91b6caea8b1e36c213`.
The original DATA.json path is now missing. Its complete object is preserved
inside tracked `experiments/osram_complete150_20261011/MANIFEST.json`.
Canonical reserialization exactly reproduces the accepted SHA256; the worker
records this recovery origin and still verifies all6598 actual file hashes.
A present but modified DATA.json never triggers fallback. No old file is replaced.
Original config, strict checkpoint load and exact historical prediction
replay establish the architecture/data bridge before switching to NEW masks.
Protected checkpoint weights are discarded before fresh initialization.
The deleted historical source directory is not reconstructed with a fabricated
manifest and is not required by the new worker.

Immutable source snapshots, effective config, data hashes, masks, optimizer/
RNG state, parameter counts, runtime and GPU UUID are retained per run.
No protected checkpoint or old output may be overwritten. The prior failed
legacy-MOSI queue is superseded, not implicitly resumed.

## Ordered scope

Run these six MOSI jobs first. Only after their finite queue completes may the
approved IEMOCAP-6 remaining-fold queue begin. GPU4 on biggpu is forbidden.
Admission checks both RAM and measured smoke VRAM; no batch/width shrinking.

Report each rate ACC/W-F1, mean eight rates, high0.5/0.6/0.7, per-seed deltas
and three-seed mean ± sample standard deviation. Do not pool rates as seeds.

## Entry point

From an immutable source snapshot on biggpu:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosi_gcnet_router_3seed_20261019.run \
  --output /data2/yb/remote_experiments/osram_mosi_gcnet_router_3seed_20261019/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpus GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
         GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
         GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

GPU order is physical7,1,3. Each carries one seed's Flat/Joint pair, at most2
training children. A real Joint smoke precedes admission. Full queue state,
PIDs, logs, checkpoints and final automatic reports live in the output root.
