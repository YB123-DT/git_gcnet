# Four direct-input mechanism trials

INTERNAL DIAGNOSTIC ONLY. Status: completed, all four runs finished 100 epochs.

## Final results (archived 2026-10-11)

W-F1 percentages; MOSI seed66, original per-rate Test-oracle BEST. These are
trained variants, not evaluation-only interventions. High missing: .5/.6/.7.

| Model | Eight-rate mean | High missing | Mean difference vs Flat (pp) |
|---|---:|---:|---:|
| Original Flat |81.068095|76.352251|0|
| Direct GATv2 |78.758614|73.262642|-2.309481|
| Direct PNA |79.685083|74.617848|-1.383012|
| Direct CWN |78.316281|72.777409|-2.751814|
| Direct Perceiver IO |78.361782|72.925621|-2.706313|

| Missing rate | GATv2 | PNA | CWN | Perceiver IO |
|---|---:|---:|---:|---:|
|0.0|86.953|86.775|87.600|87.380|
|0.1|85.121|85.704|84.472|84.584|
|0.2|80.897|81.954|80.152|80.218|
|0.3|79.906|80.253|80.123|78.927|
|0.4|77.404|78.942|75.850|77.008|
|0.5|74.391|75.425|73.593|72.864|
|0.6|72.340|73.770|71.708|72.527|
|0.7|73.057|74.658|73.031|73.386|

All four are below Flat. Together with the two initial direct trials, none of
six improve the eight-rate or high-missing mean. Best direct model is Neural
Production (79.971398/75.376807). This does not support the specific hypothesis
that simply removing the external raw evidence addition will recover gains.
Interface, parameter count and decoder initialization differences prevent a
universal causal conclusion about residual connections. No automatic expansion.

Final evidence is under `final_records/`: dispatcher, source snapshot, summary,
and per-run config/metrics/history/diagnostics/provenance. Four exit codes are 0;
provenance outputs_verified=true. Archived config/history/metrics hashes match
provenance; all six trials share eight evaluation mask hashes. Checkpoints and
selected predictions remain on biggpu and are not uploaded to Git.

Approved: CWN, GATv2, PNA, Perceiver direct evidence transforms. Original cfg84
no-JEPA MOSI seed66,100epochs, causal Memory, cyclic random missing0.0–0.7,
constantAdam1e-3, original taskMSE/batch32 and per-rate Test-oracle BEST.
No new loss/Gate/completion or Memory/query/write changes. Old XCA/NPS jobs are
independent and untouched; no rerun of historical baselines.

Final computation: `head(LN(skip(original L)+adapter([L',B',G'])))`.
No external raw evidence addition. Original Flat Adapter/Local Skip/norm/head
remain. First/empty-history rows use original inputs. InactiveGap/padding hidden
zero; original backward halves remain zero. Internal mechanism shortcuts remain.

| Switch | Actual adaptation | Module parameters | Historical residual module parameters |
|---|---|---:|---:|
| m03_gatv2_direct | Five-role64d GATv2 node updates, direct slot decoding |107968|182592|
| m05_pna_direct | Five-role64d PNA node updates, direct slot decoding |169408|244032|
| cwn_cellular_direct | Original real33head-token cell complex; direct decoded slots |362112|362112|
| perceiver_io_direct | Original128d tokenizer/eight latents/three processors; decode active token queries |732544|839872|

GATv2/PNA expose outputs before the original pooled readout, then use normal-init
Local64→256/shared Memory64→512 decoders. Their now-unused flattened pooled
readout is removed in direct instances only. Old pooled operators still execute
the same node-update and pooling equations. CWN uses TokenAdapter residual=False,
zero_decoder=False; core weights, parameter count and RNG draw order matched.
Perceiver decodes each real typed token from the same transient latent set: original
Local-derived query plus Memory token queries. Normal Local128→256/eight per-head
128→64 decoders reconstruct interface dimensions, not missing raw features.
This changes output query/readout interfaces and parameter counts; it is not a
pure parameter-matched removal-only residual ablation. No extra persistent state.

## Verification

- Tests written and observed RED for missing registration/role encoding first.
- Local22passed/5skipped +19subtests: direct semantics, first/padding/inactiveNaN,
  backward halves, finite gradients/real core update, CWN initialization/RNG,
  prior XCA and original40-method configuration regression.
- Remote CPU15passed: all four full-model checks plus prior XCA/NPS checks.
  Original shared parameter initialization and Local Skip preserved; padding
  hidden zero and predictions finite. Original task head bias is unchanged.
- Python compilation and diff whitespace checks passed. No GPU smoke.

## Historical launch and queue record

Server biggpu, physicalGPU7 UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`.
GPU4 forbidden. Remote root `/data1/yb/remote_experiments/osram_direct_input4_20261010`.
Tracked code commit `bca9ea1`; immutable source archive and per-file hashes sealed.
Dedicated tmux socket `gcnet_direct_input4`, session `direct4_seed66`.
Dispatcher PID3786007, at most two NEW concurrent runs; old XCA/NPS stay running.
Each new run reserves5000MiB plus2048MiB safety, including pending allocation
growth of its sibling. Prelaunch free13345MiB. No unrelated process termination.
GATv2 admitted PID3786474; PNA admitted PID3787532. Both verified completed epoch1,
with finite losses and no traceback in inspected logs. CWN/Perceiver pending.
Remaining approved jobs admitted automatically
when capacity permits; failures recorded without automatic repeat/tuning.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_direct_input4_20261010.dispatch \
--root /data1/yb/remote_experiments/osram_direct_input4_20261010 \
--reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66/config.json \
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Current state/PIDs/commands: DISPATCH.json; per-run log:
`runs/{method}/seed_66/train.log`. Per-run provenance saves config, snapshot,
data/version/environment/GPU and output hashes. Full recovery checkpoint/eight
BEST/eight selected predictions retained. Completion requires100epochs and
original evaluation mask hashes/artifacts verified; no auto multi-seed expansion.
