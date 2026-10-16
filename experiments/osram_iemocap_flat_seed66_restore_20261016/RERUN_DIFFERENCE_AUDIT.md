# Why the Flat seed66 rerun differs from the historical run

INTERNAL TEST-ORACLE DIAGNOSTIC. Read-only investigation on2026-10-16;
no new training, inference, hyperparameter changes or checkpoint deletion.

## Confirmed runtime mismatch

The historical corrected-mask run's `DISPATCH.json` records
`/home/qinjinghui/anaconda3/envs/adapter/bin/python` in the actual commands.
The new run uses `/data2/yb/conda_envs/cmerc-repro/bin/python`.

| Item | Historical interpreter, inspected now | New interpreter |
|---|---|---|
| Python | 3.12.12 | 3.10.13 |
| PyTorch | 2.9.1+cu128 | 1.13.1+cu117 |
| Torch CUDA build | 12.8 | 11.7 |
| cuDNN | 91002 | 8500 |
| NumPy | 2.3.0 | 1.26.4 |
| scikit-learn | 1.8.0 | 1.3.2 |
| Configured CPU threads | old dispatcher sets1 | new worker sets2 |

The old environment's versions were queried **now**: its original run did
not retain a complete package snapshot, so they are not certified historical
package versions. The different interpreter paths are directly established by
the launch records. Old four-class jobs used GPU0 and six-class jobs GPU1;
new jobs used GPU1 except six-class fold5 on GPU3. The UUID enumeration/RNG
repair is also new; it is not individually identified as the source of drift.

This is a same-protocol **cross-environment rerun**, not a demonstrated
bit-for-bit reproduction. Changing numerical backends can alter optimization
trajectories despite a common seed; it does not imply systematically higher
accuracy. No isolated same-environment comparison has measured its share of
the observed difference.

## Config and source checks

Compared all ten old/new seed66 configs. Existing task/model/training
hyperparameters match, except logical device spelling. New fields are
selection-artifact retention, explicit ACC selection and disabled/default
options: causal-GRU false, disable-local-skip false, RPCA multiplier1 with
meaningful block none. Local Skip is retained; no B6 is enabled.

Compared numerical source `b4a3138` with `785a012`. The loader, corrected
missing sampler, mixed-rate schedule, task loss and resume implementation are
unchanged. `train_epoch`, view preparation, task metrics and epoch learning
rate application are AST-identical. Added model flags remain disabled on
Flat; the original readout formula and optimizer parameter ordering remain.
Exact-artifact collection copies already evaluated outputs and adds no
model forward or random sampling. Fresh TrainingState saves/clones state
without changing optimizer updates or consuming training RNG.

This excludes an identified intentional Flat mathematical change; it does
not establish bitwise equivalence across frameworks or numerical kernels.

## Actual trajectory evidence from100 epochs, ten folds

All ten folds have identical first-epoch train ACC and all eight first-epoch
test ACCs. First-epoch train losses differ by0 to1.845e-5; subsequent discrete
train/test predictions diverge at different epochs. This supports early
small numerical differences followed by optimization divergence, rather than
an immediately different model or grossly different mask distribution.

Every retained train schedule statistic is identical at every epoch:
rate batch/conversation/valid-utterance counts, modality-element and missing
counts, realized fractions, optimizer-step and forward counts, routing and
recorded schedule hashes. Historical exact selected masks were discarded,
so these statistics do **not** prove every mask bit is identical. Do not
claim the improvement came from easier newly sampled test masks.

| Classes | Fold | First differing train ACC epoch | First differing test ACC epoch | Old eight-rate ACC | New eight-rate ACC | Difference pp |
|---|---|---|---|---|---|---|
|4|1|19|15|75.910138|75.506912|-0.403226|
|4|2|5|5|81.561584|82.416911|+0.855327|
|4|3|5|6|77.660730|78.779322|+1.118593|
|4|4|3|2|78.734239|78.697866|-0.036372|
|4|5|6|7|82.121273|82.342869|+0.221595|
|6|1|4|4|59.914421|60.988711|+1.074290|
|6|2|15|16|63.504794|64.896755|+1.391962|
|6|3|22|14|59.026450|58.739643|-0.286807|
|6|4|5|4|60.019841|61.607143|+1.587302|
|6|5|23|16|63.408811|63.824707|+0.415896|

Seven folds improve and three decline. The equal-fold ACC increase is
0.351183pp for four classes and0.836529pp for six classes. Rate0.0 scores
also differ, so merely changing the final missing-test mask cannot explain
the entire effect. Both records use100 epochs and per-rate ACC Test-best;
neither fresh-mask retest nor three-seed mean is substituted here.

## Interpretation and limits

The leading explanation is optimization trajectory drift under different
runtime environments, consistent with tiny early loss differences growing
into different later predictions. We have not isolated the responsible
kernel, CPU-thread setting, synchronization or other runtime component.
There is no evidence that recording UA/checkpoints itself improves the model.
The existing Test-best rule can select different later maxima once training
trajectories diverge, but the selection rule was not newly changed.

B6 and this new Flat use the same numerical source and cmerc-repro runtime,
so prefer that comparison when assessing B6. It still is not an exact-mask
paired experiment or a statistically significant multi-seed claim. Do not
erase the historical Flat results or attribute this rerun gain to a module.

Sources: old `DISPATCH.json`, ten old/new `config.json` and `history.json`,
old `metrics.json`, new per-fold `SUMMARY.json`, and the two committed source
trees. Output roots and immutable identities are recorded in LAUNCH.md and
the earlier B6 BASELINE_COMPARISON.md.
