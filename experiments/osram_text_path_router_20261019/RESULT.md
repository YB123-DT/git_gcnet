# Frozen conditional historical-Text decision Router: completed pilot

INTERNAL FROZEN-ANCHOR EXPERIMENT. Source `282b26a`, seed66,30epochs each.
MELD original fixed miss0.3 anchor; IEMOCAP-6 original fixed miss0.3 anchor / fold5.
No original parameter, Memory formula, Query, GCNet missing sampler or task head
was changed. Only95,439 Router parameters trained. No double view or extra loss.

## Eight-rate whole-test means (%)

| Dataset | Model | ACC | UA | W-F1 | CE |
|---|---|---:|---:|---:|---:|
| MELD | Frozen original Flat | 61.336 | 38.861 | 58.643 | 1.194451 |
| MELD | Fixed uniform probability mixture | 61.236 | 38.516 | 58.374 | 1.193871 |
| MELD | Learned Router | 61.226 | 38.542 | 58.384 | 1.193737 |
| IEMOCAP-6 fold5 | Frozen original Flat | 62.616 | 62.092 | 62.383 | 0.977890 |
| IEMOCAP-6 fold5 | Fixed uniform probability mixture | 62.169 | 61.720 | 61.943 | 0.983349 |
| IEMOCAP-6 fold5 | Learned Router | 62.631 | 62.110 | 62.400 | 0.977010 |

| Dataset | Router ΔACC pp | ΔUA pp | ΔW-F1 pp | High ΔACC pp | High ΔUA pp | Selected Router epoch |
|---|---:|---:|---:|---:|---:|---:|
| MELD | -0.110 | -0.318 | -0.259 | -0.115 | -0.424 | 1 |
| IEMOCAP-6 fold5 | +0.015 | +0.018 | +0.017 | +0.021 | +0.038 | 30 |

The F0 values are **one fixed checkpoint across eight exact masks**, not the
per-rate BEST baseline aggregate. MELD baseline61.557/37.728 is not replaced or
redefined. Original anchor selection was Test-oracle; Router choice used only
fixed validation masks (ACC first, ties lower CE), one Router for all test rates.

## What happened

- MELD: no benefit, including UA. The validation-selected epoch1 still assigns
  roughly0.385/0.352/0.263 to BG/B/G, close to uniform. Continued training did not
  improve the registered validation ACC criterion. This does not establish that
  the Router lacks optimization or that another LR/epoch would fix it.
- IEMOCAP: Router restores much of the uniform-mixture degradation by assigning
  roughly90% to BG, but the final ACC/UA change is negligible. This is not a
  meaningful success claim; only fold5/seed66 was run.
- Across the eight rate-conditioned prediction sets (not unique utterances),
  MELD corrections/harms68/91; IEMOCAP9/7. Noneligible predictions are unchanged.
- The loss and CE can slightly improve while ACC/UA do not. Do not call lower CE
  a classification improvement.
- Persistent is bypassed by construction, hence cannot improve in this design.

## Verification, cost and outputs

- 49 local unit/regression tests,21 remote Router tests passed.
- Initial GPU smoke source84c841e failed exact candidate parity due head-input
  tensor layout; preserved as failed smoke, never used for formal training.
  Corrected source passed real MELD/IEMOCAP smoke without tolerance relaxation.
- At all eight test rates, original and both eligible source-removal candidates
  reproduce prior cached logits bit-for-bit; IDs, labels, speakers and masks match.
- Independently recomputed ACC, explicit6/7-class UA and correction/harm counts;
  verified every saved prediction NPZ hash and unchanged protected anchor hashes.
- Model state digest unchanged after all training. All original parameters stayed
  frozen without gradients. Optimizer excludes them. Complete Router/Adam/RNG
  recovery reproduces the next update in regression testing.
- MELD:322.07s including setup/evaluation,338.37MiB peak allocated GPU memory,
  990 training batches /866 optimizer updates (no-eligible batches skipped).
- IEMOCAP:172.84s,803.28MiB peak allocated,120 batches /106 optimizer updates.
  Runs overlapped on healthy biggpu GPU1; timing is not an isolated throughput claim.
- Per-rate full/eligible metrics, class recall, weights and corrections/harms:
  `MELD/RESULT.md`, `IEMOCAPSix/RESULT.md`, corresponding `SUMMARY.json/METRICS.csv`.
- Remote root `/home/qinjinghui/remote_experiments/osram_text_path_router_20261019/`;
  full train histories, best_router.pt, last.pt and prediction NPZs retained there.

## Decision and next-stage boundary

The finite frozen-Router pilot does not show useful improvement. Do not expand
seeds or hyperparameter search automatically. A from-scratch one-stage comparison
would be a distinct experiment, not a reinterpretation of these results. It must
actually restore all required gradient paths and control shared Dropout, see
`ONE_STAGE_NOTE.md`. No one-stage training has been launched in this run.
