# MOSI causal eta=.6 emotion readout experiment

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

Approved: three new variants × seeds66–70, 100 epochs each. Full inherited from `/data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_N/`. Only new `osram_emotion_ablation` changes relative to that seed's configuration. Training starts from scratch, not from B2 or Full weights. Natural mask schedule, optimizer, LR=.001, batch32, H8/K32/V32/output700, eta=.6 and cyclic schedule are unchanged.

| Variant | Classification slots | Structured predictor contexts |
|---|---|---|
| Full | Local + Base + Gap | Base + Gap |
| Local-only | Local | Base + Gap |
| Local+Base | Local + Base | Base + Gap |
| Local+Gap | Local + Gap | Base + Gap |

Legacy `osram_ablation` also masks returned predictor contexts and is intentionally NOT used. New masking only zeros emotion-fusion slots after the unchanged scan. No modules or parameters are removed. Local-only still learns through auxiliary JEPA; it is not a no-memory-training model.

## Selection

For every seed AND every rate independently choose maximum Test weighted-F1 among epochs1–100, earliest tie. Eight-rate mean is calculated only AFTER those per-rate selections. Full uses identical extraction. Every epoch evaluates all eight test rates; no additional training per rate.

The existing trainer's `best.pt` saver retains its one mean-selected checkpoint as a storage artifact only; **metrics in that file are NOT used for the requested table**. `history.json` supplies the per-rate result. Not every independently selected epoch's weights are retained. This distinction has no effect on optimizer updates or reporting maxima.

## Verification before launch

- 225 model tests passed (OSRAM, write-step historical equivalence, B2 compatibility, Missing-M3, new emotion-only tests); existing PyG deprecation warning only.
- New tests confirm exact parameter tensors/RNG after initialization, strict state loading, per-write memory equality, unchanged returned contexts, actual classification hidden change with unchanged regression AND contrastive predictions, finite backward, padding, config/CLI defaults and serialization, legacy exclusivity.
- Six runner/summary standard-library tests pass: exact one-field clone, locked settings, complete100epoch/rate histories, independent maxima and earliest ties, mask matching, pending suppression, sample std/seed-first aggregate counts and LF output.
- `git diff --check` and Python syntax compilation pass. No separate 1-epoch training smoke was launched.

## Execution

`python experiments/osram_causal_readout_20260910/launch.py --launch`

GPU2 seed66, GPU3 seed67, GPU5 seed68, GPU6 seed69, GPU7 seed70; each runs all three variants concurrently. Busy0/1 and unhealthy4 avoided. Launcher refuses duplicate QUEUE.json; workers refuse existing output directories. Remote root `/data2/yb/remote_experiments/osram_causal_readout_20260910/` holds logs and provenance. On completion launcher runs `summarize.py`, requiring all15 new +5 inherited runs and identical per-seed evaluation mask hashes before reporting complete means.

Status at code handoff: verified and ready to launch; live state is QUEUE.json, not this static document. No new F1 results are claimed here.
