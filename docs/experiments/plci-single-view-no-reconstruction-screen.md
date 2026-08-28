# Single-View PLCI Without Reconstruction — IEMOCAP-6 Screen

## Question

Does replacing the inherited raw-feature reconstruction objective with the
Single-View JEPA objective improve the existing Single-View PLCI result?

## Locked Comparison

- Dataset: `IEMOCAPSix`
- Fold: `5`
- Missing rates: `0.5`, `0.7`
- Seeds: `66`, `67`, `68`, `69`, `70`
- New jobs: `10`
- GPUs: `0`, `1`, `2`, at most three jobs per GPU
- Epochs: `100`
- Reconstruction-enabled Single-View jobs rerun: `0`
- Original jobs rerun: `0`

The new and inherited commands are identical after removing their output
directories and the inherited command's `--loss-recon` flag. The model still
instantiates the unused linear reconstruction head so model construction,
parameter initialization, and RNG behavior remain paired. This screen changes
only whether raw-feature reconstruction contributes to the training loss.

The compared objectives are:

```text
inherited: emotion + raw-feature reconstruction + JEPA
new:       emotion + JEPA
```

## Remote Execution

- Code: `/data2/yb/paper/GCNet_TPAMI_single_view_dev`
- Python: `/data2/yb/reproduction_envs/gcnet-official/bin/python`
- Output: `/data2/yb/paper/experiments/plci_single_view_norecon_iemocap6_screen_20260828`
- Launcher log: `/data2/yb/paper/plci_single_view_norecon_launcher_20260828.log`

```bash
cd /data2/yb/paper/GCNet_TPAMI_single_view_dev
/data2/yb/reproduction_envs/gcnet-official/bin/python \
  scripts/run_plci_single_view_iemocap6.py \
  --output-root /data2/yb/paper/experiments/plci_single_view_norecon_iemocap6_screen_20260828 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python \
  --gpus 0 1 2 \
  --jobs-per-gpu 3 \
  --epochs 100 \
  --missing-rates 0.5 0.7 \
  --seeds 66 67 68 69 70 \
  --no-loss-recon
```

## Startup Record

The first launcher reached argument validation and exited before reading data
because an older PLCI guard required `--loss-recon` for both architectures.
The guard now remains active for Dual-View `plci` but permits the explicitly
selected `plci-single` replacement experiment. The pre-training failure logs
are preserved at
`/data2/yb/paper/experiments/plci_single_view_norecon_iemocap6_screen_20260828_startup_failed_validator`.

