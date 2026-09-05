# MOSI cyclic vs. stratified mixed-rate training

## Question

Does the way that missing rates are assigned during training matter when the
model and evaluation masks are held fixed?

- **Cyclic:** one missing rate is used by the whole batch; the rate advances
  through `0.0, 0.1, ..., 0.7` across batches.
- **Stratified:** conversations in the same batch receive different rates,
  with the batch assignments kept as balanced as possible across the eight
  rates.

This is a protocol comparison only. No model module, feature, graph, loss,
optimizer, or evaluation mask was changed.

## Paired setup

| Item | Value |
|---|---|
| Dataset | CMU-MOSI, regression |
| Seeds | 66, 67, 68, 69, 70 |
| Fold | 1 |
| Epochs | 100 |
| Batch size | 32 |
| Backbone | GCNet, `hidden=200`, `windowp=2`, `windowf=2`, `time_attention=False` |
| Missing-M3 | Text-anchor residual, slot representation |
| Natural training schedule | cyclic or stratified (the only intervention) |
| Evaluation | all eight rates: 0.0--0.7 |
| Selection policy | **Test-oracle diagnostic**, independently select the best epoch for each rate |
| Validation | not used for this diagnostic, as requested |
| Evaluation masks | identical fixed mask bank for every paired seed/rate |

The comparison is therefore not a formal validation-selected benchmark. It is
the requested diagnostic for separating the effect of cyclic versus stratified
training-rate assignment. The per-rate Test-oracle selection also means that a
rate can use a different epoch from another rate.

## Per-rate Test-oracle W-F1 (%)

Each cell is `best W-F1 @ epoch` for that seed and rate. The mean is over the
five seeds. `Delta` is stratified minus cyclic in percentage points.

| Missing rate | Cyclic (66/67/68/69/70) | Cyclic mean | Stratified (66/67/68/69/70) | Stratified mean | Delta |
|---:|---|---:|---|---:|---:|
| 0.0 | 86.7346@63 / 87.1531@61 / 86.7278@53 / 85.8733@60 / 87.0824@78 | 86.7142 | 85.2239@89 / 86.2092@53 / 86.0113@49 / 85.8733@71 / 85.8062@57 | 85.8248 | -0.8895 |
| 0.1 | 85.5350@63 / 86.2437@65 / 84.7703@53 / 84.0596@71 / 85.3795@59 | 85.1976 | 83.5366@72 / 84.8147@53 / 84.1746@61 / 84.0421@74 / 84.9628@57 | 84.3061 | -0.8915 |
| 0.2 | 82.8086@63 / 84.2326@61 / 82.6302@55 / 81.1334@60 / 81.2454@71 | 82.4100 | 82.0368@78 / 83.2058@53 / 82.4471@80 / 81.2200@68 / 80.5226@64 | 81.8865 | -0.5236 |
| 0.3 | 80.9752@63 / 81.9066@61 / 80.1990@41 / 81.4704@46 / 80.8305@47 | 81.0763 | 80.5304@94 / 80.4772@56 / 82.0145@81 / 81.3853@74 / 79.6123@55 | 80.8039 | -0.2724 |
| 0.4 | 80.2100@63 / 76.5151@64 / 79.2903@54 / 79.4719@71 / 79.9769@53 | 79.0929 | 79.1264@77 / 75.8836@56 / 81.7541@61 / 78.4008@62 / 82.4261@54 | 79.5182 | **+0.4253** |
| 0.5 | 77.3785@69 / 76.6525@46 / 77.4041@52 / 76.6311@71 / 76.5151@95 | 76.9163 | 76.7784@77 / 76.3578@47 / 77.4675@83 / 76.0340@68 / 77.4940@75 | 76.8263 | -0.0899 |
| 0.6 | 76.3542@69 / 76.9823@66 / 75.9525@82 / 76.2093@73 / 76.4430@53 | 76.3883 | 76.2077@52 / 76.1558@53 / 77.6842@93 / 75.5977@75 / 76.5841@54 | 76.4459 | **+0.0576** |
| 0.7 | 76.0612@69 / 75.1338@64 / 70.7426@54 / 76.1048@71 / 76.0632@53 | 74.8211 | 75.1219@63 / 74.1857@65 / 71.0365@62 / 75.6753@62 / 74.9170@54 | 74.1873 | -0.6338 |
| **Eight-rate mean** |  | **80.3271** |  | **79.9749** | **-0.3522** |

## Paired interpretation

- Stratified is higher at only rates **0.4 and 0.6**, and only by `+0.4253`
  and `+0.0576` points, respectively.
- Cyclic is higher at the other six rates, including complete input (`0.0`)
  by `+0.8895` points and `0.7` by `+0.6338` points.
- At the seed level, stratified minus cyclic eight-rate deltas are:
  `seed 66=-0.9369`, `67=-0.9412`, `68=+0.6091`, `69=-0.3407`,
  `70=-0.1514` points. Only **1/5** seeds is positive.
- Standard deviation of the five seed-level eight-rate means is `0.4160`
  points for cyclic and `0.3091` points for stratified. Stratified is somewhat
  less variable, but its mean is lower.

## Integrity checks

- Five stratified runs reached epoch 100, produced `metrics.json`, eight NPZ
  prediction files, and no `Traceback`/`ERROR` in the logs.
- For every seed, all eight evaluation-mask SHA256 values exactly match the
  paired cyclic run.
- Both runs use the same `text-anchor-residual`, slot, GraphConv-second-layer,
  no-postgraph-BiLSTM-ablation configuration; only `train_rate_mode` differs.
- The trainer's `metrics.json` still records its single global eight-rate-mean
  checkpoint. The table above intentionally recomputes the requested
  **independent per-rate Test-oracle maxima from `history.json`**; the NPZ files
  are not being misrepresented as eight separately saved checkpoints.

## Conclusion

Under this exact paired MOSI diagnostic, **cyclic performs better than
stratified by 0.3522 percentage points in the eight-rate mean**. Stratified
does not provide a reliable improvement: it wins 2/8 rates by mean, and only
1/5 seeds has a positive overall delta. It may reduce seed variance slightly,
but the current evidence does not justify replacing cyclic training with
stratified training.

This conclusion is limited to the current Text-anchor Missing-M3 model and
the Test-oracle selection policy. A validation-selected formal comparison
would be a separate experiment and was deliberately not run here.

