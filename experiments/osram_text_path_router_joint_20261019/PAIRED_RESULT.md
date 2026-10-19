# One-stage historical Text-path probability Router

INTERNAL SINGLE-SEED EXPLORATION

MELD seed66; IEMOCAP-6 seed66/fold5. From-scratch matched Flat and Joint jobs.
One checkpoint selected by validation mean eight-rate ACC (ties lower CE), then replayed
on the same historical exact GCNet test masks. Historical test-guided research and mask
provenance preclude an independent-test paper claim. Not per-rate Test-best scores.

| Dataset | Model | Selected epoch | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MELD | flat | 37 | 60.4358 | 37.5037 | 57.7353 | 56.4751 | 31.6796 | 52.3435 |
| MELD | joint | 37 | 60.3688 | 37.4310 | 57.6475 | 56.3218 | 31.5384 | 52.1630 |
| IEMOCAPSix | flat | 75 | 58.8417 | 59.6742 | 59.2588 | 57.0959 | 57.7867 | 57.5313 |
| IEMOCAPSix | joint | 94 | 59.7659 | 59.6348 | 59.9706 | 57.5478 | 57.7443 | 57.6843 |

## Interpretation safeguards

- Joint vs separate Flat measures the full training-method change.
- Within Joint, mixture vs its own full candidate measures the learned readout contribution.
- Uniform Router initialization is not a Flat identity on eligible samples.
- Eligibility is current Text missing AND strictly-past Text seen; no labels enter the Router.
- Source terms are joint Value-slot contributions, not pure semantic modality attribution.
- No automatic expansion to other seeds/folds or hyperparameter search.
