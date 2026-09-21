# Three-dataset M3 pretraining

This experiment trains one utterance-level bank shared by CMU-MOSI, CMU-MOSEI,
and IEMOCAP.  It deliberately stops before OSRAM/GCNet classification: the only
trainable components are the three `ModalityProjector`s and the shared
`SourceOnlyM3Predictor`; targets are produced by frozen EMA copies.

Each valid utterance receives one of the six non-empty source patterns
`A/T/V/AT/AV/TV`.  Missing targets are supervised with the repository's existing
`missing_m3_loss` (SmoothL1 plus symmetric InfoNCE).  A multi-source target is
the mean of its source-specific predictions, matching the existing predictor.

The loader keeps train and validation separate (`strict` protocol by default).
IEMOCAP uses one selected LOSO fold (`--iemocap-fold`, default 5); its test
loader is never iterated by this experiment.  Shared projectors require the three datasets to expose the
same `(audio_dim, text_dim, visual_dim)` tuple; a mismatch fails explicitly
rather than silently adding dataset-specific adapters.

## Smoke test

The smoke test is synthetic and does not touch data or start a long run:

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m experiments.m3_pretrain_three_datasets_20260921.run --smoke
```

## Explicit training usage

The CLI never starts a real run unless `--run` and all three feature-root
entries are supplied.  For example (with the repository's feature names):

```bash
python -m experiments.m3_pretrain_three_datasets_20260921.run --run \
  --feature-root CMUMOSI /path/mosi/audio /path/mosi/text /path/mosi/visual \
  --feature-root CMUMOSEI /path/mosei/audio /path/mosei/text /path/mosei/visual \
  --feature-root IEMOCAPSix /path/iemocap/audio /path/iemocap/text /path/iemocap/visual
```

The same functionality is available through `build_joint_loaders` and
`train_joint` for experiment drivers.  A checkpoint is saved as:

```python
{
    "model_config": {"dimensions": dims, "latent_dim": 256},
    "model_state": model.state_dict(),
}
```

run `evaluate.py` with one `--feature-root DATASET AUDIO TEXT VISUAL` option for
each dataset.  It reports, separately for train and validation, centered cosine,
prediction/target standard deviation, prediction MSE, and the gap against the
train-target-mean baseline for all six single/double source patterns and all
nine directed routes.  The requested `A->T`, `V->T`, and `AV->T` routes are
therefore directly visible in the JSON output.

The dataset label root remains controlled by the repository's existing
`GCNET_DATASET_ROOT`/`config.py` convention.
