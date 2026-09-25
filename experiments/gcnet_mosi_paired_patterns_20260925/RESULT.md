# MOSI: original GCNet reconstruction ablation, fixed modality patterns

Completed on remote `biggpu` / `user22`. These are original GCNet models,
not OSRAM, MMoE, JEPA or pretrained latent-completion models.

## Results

Test Nonzero binary weighted F1 (%), mean ± sample SD over seeds 66/67/68.
Delta is **without reconstruction minus with reconstruction**, in percentage
points. No seed or low-scoring result was excluded.

| Available modalities | With reconstruction | Without reconstruction | Delta |
|---|---:|---:|---:|
| A | 40.594 ± 9.502 | 52.691 ± 10.956 | +12.097 |
| T | 76.777 ± 2.322 | 79.668 ± 4.873 | +2.891 |
| V | 49.029 ± 9.181 | 54.421 ± 0.713 | +5.392 |
| AT | 84.231 ± 0.273 | 83.490 ± 0.631 | −0.742 |
| AV | 56.695 ± 1.421 | 55.690 ± 5.280 | −1.005 |
| TV | 83.182 ± 0.921 | 84.373 ± 0.688 | +1.191 |

T = L = Text/Language. Scores above come from the final unified two-arm
evaluation, not the earlier no-reconstruction-only preliminary evaluation.

## Protocol and scope

- Official MOSI train/validation/test splits: 1,284 / 229 / 686 utterances.
  Nonzero binary metrics exclude neutral labels (656 test utterances).
- Six new reconstruction controls: rates .3/.7 × three seeds, 100 epochs each.
  Without-reconstruction models reuse the completed 20260924 runs.
- With reconstruction: original `linear_rec` and original `MaskedReconLoss`
  alongside task MSE. Without reconstruction: no prediction head and task MSE
  only. No predicted-feature reinjection in either classification path.
- Shared settings: LSTM, hidden 200, graph hidden 100, windows 2/2, dropout .5,
  no time attention, batch 32, Adam lr .001, weight decay 1e-5, no gradient clipping.
  Fixed upstream wav2vec-large-c / DeBERTa-large-4 / MA-Net utterance features.
- Earliest maximum fixed-mask validation Nonzero W-F1 selects `best.pt`.
  Test scores do not select checkpoints. `last.pt` retains optimizer/RNG state.
- Singles A/T/V use the .7-trained best checkpoint. Pairs AT/AV/TV use the
  .3-trained best checkpoint. This mapping was fixed before testing.
- Original nominal .7 masking leaves exactly one modality per utterance
  (actual missing fraction 2/3). Rate .3 is random missing training, not fixed
  two-modality training.
- Evaluation-only intervention: all utterances of each test conversation are
  given the same specified availability pattern. No separate A/T/V/etc. training;
  no natural-missing subgroup analysis; no pattern-specific checkpoint selection.
- Verified 100 epochs of training and validation mask hashes and conversation
  order match across each of the six paired runs. Final evaluation also checks
  labels, conversation order, masks and counts match between arms.
- Recomputed all 36 stored fixed-pattern scores from remote prediction NPZs.
  Verified six complete 100-epoch control histories, validation-selected epochs,
  and existence of best/last checkpoints.

## Interpretation and limits

Removing reconstruction improves the three single-modality means in this
experiment; pair-modality changes are mixed. A-only has substantial seed
variation. This is not evidence of a statistically significant gain, universal
harm from reconstruction, or the information ceiling of audio/visual features.
Fixed-pattern diagnostic results must not be substituted for random-missing-rate
benchmark results. The historical 20260819 baseline used a different evaluation
mask protocol and is not the control in this table.

## Evidence and retained weights

- `remote_summary.json`: full-precision means, sample SDs and deltas.
- `results/*_seed_*.json`: all 36 individual metrics, selected epoch,
  checkpoint location/hash, mask hash, label hash and conversation order.
- `pairing_audit.json`: six successful 100-epoch pairing checks.
- Reconstruction best/last checkpoints remain at
  `/data2/yb/remote_experiments/gcnet_mosi_paired_patterns_20260925/code/experiments/gcnet_mosi_paired_patterns_20260925/runs/`.
- Without-reconstruction checkpoints remain at
  `/data2/yb/remote_experiments/gcnet_mosi_no_reconstruction_20260924/code/experiments/gcnet_mosi_no_reconstruction_20260924/runs/`.

This upload contains result documentation and small metric JSONs only, not
weights, features, datasets or prediction arrays. Training/evaluation source
snapshots and detailed logs remain in the corresponding remote code directories.
