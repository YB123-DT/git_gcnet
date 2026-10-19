# One-stage Text-source Router: seeds66/67/68

INTERNAL EXPLORATORY EXPERIMENT

Seed66 reused without retraining. Seed67/68 add matched fresh Flat and Joint jobs.
MELD official split; IEMOCAP-6 fold5 only. Fixed split and evaluation masks across seeds.
One validation-selected checkpoint per job, not per-rate Test-best.

| Dataset | Seed | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MELD | 66 | flat | 60.4358 | 37.5037 | 57.7353 | 56.4751 | 31.6796 | 52.3435 |
| MELD | 66 | joint | 60.3688 | 37.4310 | 57.6475 | 56.3218 | 31.5384 | 52.1630 |
| MELD | 67 | flat | 60.6944 | 38.0023 | 57.9286 | 56.7050 | 32.1328 | 52.6218 |
| MELD | 67 | joint | 60.6705 | 37.9865 | 57.8967 | 56.6922 | 32.1691 | 52.6057 |
| MELD | 68 | flat | 60.4071 | 38.6287 | 57.6864 | 56.5900 | 31.9011 | 52.4724 |
| MELD | 68 | joint | 60.3975 | 38.6865 | 57.6879 | 56.5900 | 31.8910 | 52.4677 |
| IEMOCAPSix | 66 | flat | 58.8417 | 59.6742 | 59.2588 | 57.0959 | 57.7867 | 57.5313 |
| IEMOCAPSix | 66 | joint | 59.7659 | 59.6348 | 59.9706 | 57.5478 | 57.7443 | 57.6843 |
| IEMOCAPSix | 67 | flat | 55.6916 | 59.1733 | 54.8029 | 53.6455 | 56.8284 | 52.5824 |
| IEMOCAPSix | 67 | joint | 60.0277 | 58.2591 | 60.2752 | 57.4245 | 55.5910 | 57.5714 |
| IEMOCAPSix | 68 | flat | 56.0613 | 55.2618 | 55.9538 | 53.6044 | 52.3065 | 53.4444 |
| IEMOCAPSix | 68 | joint | 58.5105 | 58.3298 | 58.9180 | 56.6646 | 55.7805 | 57.0941 |

## Three-seed mean ± sample standard deviation

| Dataset | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| MELD | flat | 60.5125 ± 0.1583 | 38.0449 ± 0.5637 | 57.7834 ± 0.1281 | 56.5900 ± 0.1149 | 31.9045 ± 0.2266 | 52.4792 ± 0.1392 |
| MELD | joint | 60.4789 ± 0.1665 | 38.0347 ± 0.6292 | 57.7440 ± 0.1337 | 56.5347 ± 0.1913 | 31.8662 ± 0.3161 | 52.4121 ± 0.2265 |
| IEMOCAPSix | flat | 56.8649 ± 1.7219 | 58.0364 ± 2.4159 | 56.6718 ± 2.3131 | 54.7820 ± 2.0041 | 55.6405 ± 2.9268 | 54.5194 ± 2.6437 |
| IEMOCAPSix | joint | 59.4347 ± 0.8110 | 58.7412 ± 0.7747 | 59.7213 ± 0.7121 | 57.2123 ± 0.4783 | 56.3719 ± 1.1923 | 57.4499 ± 0.3133 |

No independent-test or multi-fold claim. Do not pool eight masks as eight independent seeds.
No additional methods, hyperparameter search or automatic expansion are authorized.
