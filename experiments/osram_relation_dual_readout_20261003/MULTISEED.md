# Authorized three-seed extension

INTERNAL DIAGNOSTIC ONLY

The user requested multi-seed checking after the seed66 negative result. Reuse
completed seed66; add seeds67/68 only, each from scratch for100 epochs. Keep the
same 0.5 base + 0.5 full objective and original cfg84 protocol. No hyperparameter
search, new modules or repeated seed66 training.

Status: completed. All three seeds have100 history entries and complete provenance.
The16 additional BEST checkpoints for seeds67/68 remain on biggpu.

## Results

W-F1 (%), per-rate BEST Test-oracle; differences are percentage points.
Each seed is compared with its matching original Flat seed, not the5-seed mean.

|Seed|Flat8-rate|Dual8-rate|Delta|Flat high|Dual high|Delta high|
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.649|-0.419|76.352|75.716|-0.636|
|67|80.556|80.519|-0.037|75.990|76.160|+0.170|
|68|80.053|79.778|-0.275|74.440|74.222|-0.218|
|Mean|80.559|80.315|-0.244|75.594|75.366|-0.228|

Across-seed sample SD (ddof=1): Flat8-rate0.507, Dual8-rate0.470;
Flat high1.016, Dual high1.015. Paired delta SD:0.193 overall,0.403 high.
All three overall means decline; high-missing improves only for seed67.
No robust improvement or formal significance claim is supported.

|Rate|Flat3-seed mean|Dual3-seed mean|Delta|
|---|---:|---:|---:|
|0.0|88.419|87.691|-0.728|
|0.1|85.845|85.875|+0.030|
|0.2|83.431|83.124|-0.307|
|0.3|80.999|80.698|-0.301|
|0.4|78.996|79.037|+0.041|
|0.5|77.327|77.183|-0.143|
|0.6|75.848|75.396|-0.453|
|0.7|73.607|73.518|-0.089|

Fixed four-cell means (same original Local-only0.25 grouping, no new grouping):

|Boundary|Adjacent polarity|Flat|Dual|Delta|
|---|---|---:|---:|---:|
|near|same|80.470|77.918|-2.551|
|near|opposite|42.048|51.179|+9.130|
|far|same|88.875|88.142|-0.732|
|far|opposite|73.648|73.405|-0.243|

Near/opposite improves in every seed, but near/same declines in every seed and
aggregate W-F1 declines. This is not evidence that the model detects emotion shifts.
Full-set corrections/harms total768/808 over15,744 seed-rate exposures, not independent
utterances. W-F1 is computed separately for each seed/rate before macro averaging.

Offline validation reused the existing analyze.py for seeds67/68, saved under
analysis_seed_67/ and analysis_seed_68/; seed66 remains in analysis/.
The scripts checked canonical sample alignment, labels and availability against
the original audited Flat predictions and recomputed W-F1 against saved metrics.
No additional training/inference was performed for this analysis. Results/configs,
history and provenance are retained under results/seed_67 and results/seed_68.
No further experiments are launched; the current variant is not recommended as
a replacement for Flat on these internal results.

|Seed|PID|Code commit|
|---|---:|---|
|67|733120|14bdda8|
|68|757004|14bdda8|

GPU UUID: GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
Remote root: /data2/yb/remote_experiments/osram_relation_dual_readout_multiseed_20261003.
An isolated code/ snapshot is used; seed_67/ and seed_68/ hold separate outputs.
Commands and PIDs are recorded in launch_seed_67.json / launch_seed_68.json;
logs are train_seed_67.log / train_seed_68.log. Training uses the previously recorded
command with --seed67 or --seed68 (separate CLI arguments), --dual-readout and
--commit14bdda8. The runner reads the corresponding original Flat seed config,
validates it, and checks the eight evaluation mask hashes on completion.

Launcher verification: all three supported seed configurations preserve baseline
fields; mismatched requested/reference seeds are rejected. No model code changed.
Live remote processes confirmed for both runs; this is not completion evidence.

After completion compare each seed with its paired Flat baseline and report all
three seeds, per-rate and8-rate/high-missing mean. Per-rate BEST remains Test-oracle;
do not present these internal comparisons as validation-selected paper results.
