# P1 frozen matched history-content intervention

No model or training change. Uses the existing P0 frozen causal, emotion-only,
no-JEPA Flat checkpoint evaluator for IEMOCAP4, IEMOCAP6 and MELD. Source
Test-oracle selection remains an internal-diagnostic limitation.

For every target and A/T/V modality, select the latest **strictly prior observed
utterance with the target's speaker**. Hold this source position fixed in both
arms. Replace only that modality's input feature slice with a distinct strictly
prior observed donor of the same modality:

- `same_speaker`: another utterance with the target speaker (content/time shuffle).
- `other_speaker`: an utterance with a different target speaker, minimizing
  source/donor lag mismatch; ties prefer the most recent donor.

The same nearest-lag rule applies to both arms. Selection takes availability,
real scalar speaker IDs (including MELD IDs 0–8), and target position only;
labels and predictions never enter selection. No source/donor fallback changes
the source to improve coverage. Missing-source and missing-donor cases are saved.
Observed content may be identical; those cases remain in the diagnostic without
selection based on features. No perturbation of modality availability occurs.

Availability trajectory, current/future inputs, qmask, umask, labels, model and
checkpoint remain fixed. Each current Local vector is checked at rtol/atol
2e-5, with exact equality and maximum absolute error recorded separately.
Replacement conversations are batched, and repeated source/donor/modality jobs
are deduplicated across targets. `--chunk-size` adjusts inference concurrency,
without changing the original loader batch size or source protocol.

`delta_ce = CE(replaced) - CE(original)`: positive means harmful replacement.
`true_probability_delta = P_true(replaced) - P_true(original)`: positive means
improved probability. Correction/harm compare original→replaced classifications.
Both arms report source/donor positions, lags, mismatch, speaker relations,
current Local and availability checks, and coverage by modality and missing rate.
Exact-lag other-speaker matches cannot occur at distinct positions within one
conversation; mismatch is explicitly retained rather than claiming exact matching.

Run from the repository root:

```bash
python -m experiments.osram_history_content_intervention_20261014.dispatch \
  --iemocap-source-root /path/to/iemocap_flat_sources \
  --meld-source-root /path/to/meld_flat_sources \
  --dataset-root /path/to/dataset --meld-pickle /path/to/source_meld.pkl \
  --output-root /new/p1_outputs --workers 3 --chunk-size 16 --device cuda
python -m experiments.osram_history_content_intervention_20261014.summarize \
  --input-root /new/p1_outputs --output /new/p1_summary
```

Default dispatch includes 2 IEMOCAP tasks × 3 seeds × 5 folds = 30 sources plus
3 MELD sources, each at rates 0.0–0.7. Source paths follow P0 dispatch. All
sources are preflighted before outputs or subprocesses are created. Output
directories must be new. Honor server ownership and GPU restrictions before
launch; this code does not reserve a GPU or train a model.

Per rate, `p0_utterances_miss_*.npz` preserves the original P0 replay;
`interventions_miss_*.npz` contains eligible paired rows;
`skipped_miss_*.json` contains every unavailable target/modality/arm.
`SUMMARY.json` records hashes, environment, checkpoints, and invariant checks.
Aggregation rechecks selection against original availability/speakers, effect
values, source labels/logits, and exhaustive eligible/skip coverage. It pools
folds within each seed before computing across-seed mean/sample standard
deviation. Datasets and missing rates remain separate; missing folds fail by
default, with explicitly marked `--allow-partial` available for inspection.
Correlated utterances do not support independent-sample significance claims.

```bash
python -m unittest discover -s tests -p 'test_osram_history_content_intervention.py' -v
```
