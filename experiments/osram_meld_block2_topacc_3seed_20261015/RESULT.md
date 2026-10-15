# MELD top Test-best ACC configurations: three seeds

INTERNAL TEST-ORACLE THREE-SEED REPLICATION; seed66 selected configurations

| Configuration | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| B6_trial16 | 66 | 60.915 | 35.767 | 57.175 |
| B6_trial16 | 67 | 60.642 | 35.710 | 56.060 |
| B6_trial16 | 68 | 60.881 | 35.464 | 57.140 |
| B6_trial16 | mean ± sample SD | 60.813 ± 0.149 | 35.647 ± 0.161 | 56.792 ± 0.634 |
| B6_trial13 | 66 | 60.910 | 34.807 | 56.621 |
| B6_trial13 | 67 | 60.407 | 36.339 | 57.146 |
| B6_trial13 | 68 | 60.757 | 36.486 | 57.068 |
| B6_trial13 | mean ± sample SD | 60.691 ± 0.258 | 35.877 ± 0.930 | 56.945 ± 0.283 |
| B6_trial12 | 66 | 60.891 | 34.843 | 56.633 |
| B6_trial12 | 67 | 60.838 | 36.799 | 57.028 |
| B6_trial12 | 68 | 60.388 | 36.359 | 56.757 |
| B6_trial12 | mean ± sample SD | 60.706 ± 0.276 | 36.000 ± 1.026 | 56.806 ± 0.202 |

Per-rate and high-missing results and full source/configuration provenance are in SUMMARY.csv/JSON.
No new seed66 training, architecture change, validation selection, or additional hyperparameter search.
Seed66 was used for configuration search; this three-seed mean is not independent test evidence.
Original seed66 source is 7abb855. Replication changes only an explicit seed admission guard and the launcher; numerical model/trainer/sampler source hashes must match.

## Completion and scope

All six additional seed67/68 runs completed 50 epochs. The requested Test-best
results above were recovered from each checkpoint's exact selected-epoch
prediction artifact, without new training or inference. Artifact hashes, selected
ACC, utterance identity sets, per-ID labels/speakers, and legal endpoint masks
were checked. Training source: `ab3d7e6`; CPU recovery source: `81039a1`.

The original workers failed only during an additional positional seed66-mask
replay after training. That cross-seed auxiliary replay is **not completed** and
supplies no results here. Original outputs/errors remain untouched; see
`POSTPROCESS_FIX.md` and `RECOVERY.json`.

Remote evidence:
`/home/qinjinghui/remote_experiments/osram_meld_block2_topacc_3seed_20261015/testbest_recovery_v1/`.

Trial16 has the highest three-seed ACC of these configurations. Against the
historical accepted Flat reference (ACC60.693 / UA36.255 / W-F1 57.045), it has
approximately +0.120 / -0.608 / -0.253 percentage points, respectively. That
historical reference is not a new same-snapshot, common-mask paired control.
These results do not demonstrate a joint ACC/UA improvement or establish that
the second block is effective. No further expansion is authorized by this report.
