Session 5 ONLY: three seeds, eight missing rates; not five-fold cross-validation.
Test-oracle selection has optimistic test-selection bias. Current runs use retained OSRAM dynamic sampling; MR0 all-visible results are reused. ComP legacy results retain their original provenance. Realized masks are not sample-paired across methods.

| Dataset | Model | Configs | ACC | UA | W-F1 | Mask |
|---|---|---:|---:|---:|---:|---|
| IEMOCAPFour | MMIN | 3/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPFour | MoMKE | 3/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPFour | GCNet | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPFour | SDR-GNN | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPFour | ComP | 24/24 | 81.131 ± 0.206 | 80.092 ± 0.074 | 80.916 ± 0.228 | legacy_dynamic |
| IEMOCAPSix | MMIN | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPSix | MoMKE | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPSix | GCNet | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPSix | SDR-GNN | 0/24 | — | — | — | dynamic_current_MR0_reused |
| IEMOCAPSix | ComP | 24/24 | 59.430 ± 0.118 | 58.697 ± 0.216 | 59.355 ± 0.086 | legacy_dynamic |
