# IEMOCAP-6 cyclic Signature launch

Status: formal training is **running**, not completed. Launch/smoke success is
not a task-performance result.

- Server: `ssh biggpu` (`user22`).
- Numerical source: `21e46f2ecd092345818659694db8ff372672c35e`.
- Snapshot: `/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2`.
- Output: `/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1`.
- Controller PID: `4165163`.
- tmux: `signature_iemocap_cyclic_20261025`.
- Controller log: `/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/controller_v1.log`.
- Environment: `/data2/yb/conda_envs/cmerc-repro/bin/python`,Torch1.13.1+cu117,
  NumPy1.26.4,sklearn1.3.2.

## Authorized finite scope

IEMOCAPSix only,seeds66/67/68,allfive folds,all8 rates:120 separate frozen-parent
Signature branches. Each trains100epochs at constant branchLR3e-4,batch32,
Adam WD1e-5,original CE. Original Flat parents trained atLR1e-3/100epochs and
remain frozen. Original GCNet missing and exact selected test masks are retained.
Select earliest strict current-rate Test-best ACC; UA/WF1/CE use the same epoch.
Internal Test-oracle exploration only. No four-class run,HPO or baseline retrain.

Physical GPU0/1/2/3 are UUID-bound with up to6taskworkers percard,24total,
subject to measured resource admission. UUID map is in README.md and live QUEUE.
GPU4 forbidden; unrelated GPU5/6/7 tasks untouched. Initial verified formal
snapshot had9workers running,111pending,no errors; admission was continuing.
Allremaining authorized tasks automatically fill released slots.

## Verification completed before formal admission

- Read-only all15 seed-fold /120checkpoint /120prediction /120mask audit passed,
  recorded in PRECHECK.json. No protected file missing or modified.
-53 focused unit/regression checks passed in local cmerc-xp-repro and actual
  remote cmerc-repro. Optional torch-sparse warning remains; Signature does not
  use that graph extension and no dependency was changed.
- Real seed66/fold5/miss.7 smoke passed two optimizer updates,original logits
  bit-exact replay (maximum absolute error0),initial branch prediction equality,
  finite/nonzero branchgradients,frozen parent/no-grad and selected branch reload.
- Smoke measured470.833MiB PyTorch allocated peak,2.616GiB peak RSS and20.574s
  including setup/data/checkpoint/mask replay. These are not full training costs;
  admission also checks actual free VRAM and reserves4096MiB.
- Same unchanged SignatureBlock has281856trainableparameters. Original model
  parameters are referenced by their protected hashes,not copied into new BEST.

The initial83eb544 source was not launched. The final21e46f2 preparation handles
seed66's older environment record honestly: Torch/NumPy match each oldsummary,
the retained launcher proves cmerc executable,and old sklearn version remains
explicitly unrecorded. Current runtime requirements and exact replay were not
relaxed. This provenance limitation is stored in every affected run identity.

## Persistence and results

Live `execution_v1/QUEUE.json` retains commands,PIDs,UUIDs,logs and statuses.
Each run stores effective config,source/parent/mask identities,all epoch metrics,
selected BEST/predictions,and full branch/Adam/epoch/RNG last.pt. Explicit resume
requires compatible identity and does not duplicate surviving children.

Final RESULT.md/JSON/CSV is automatic after120 validcompletedruns. The collector
independently checks selected prediction metrics,mask/checkpoint integrity and
same-epoch ACC selection. It reports foldmeans and completefivefold seedmeans,
with pooled out-of-fold metrics labelled separately. Incomplete seeds never
receive an invented fivefold mean. No weights,feature caches or large predictions
are uploaded to GitHub; preserve them remotely for reproducibility.
