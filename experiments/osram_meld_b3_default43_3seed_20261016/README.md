# MELD default B3 4/3 three seeds

INTERNAL TEST-ORACLE THREE-SEED REPLICATION. Exactly two new runs:67/68.
Reuse completed seed66,ACC60.756705/UA37.357830/W-F1 57.395504.

All numerical training code and worker are the original immutable737fdb1:
`/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1`.
Only seed changes. Default128dim/4heads/FFN256,4tokens/3rounds; Flat1600 and
raw Local Skip; CHM unchanged; batch32,50epochs,Adam LR0.001,WD0.00001,dropout0.5,
cyclic0.0–0.7,`gcnet-random-mask-v1`,per-rate Test-best ACC selection.
UA is seven-class macro recall from the same selected-epoch predictions.

Run on biggpu physical GPU3,UUID`GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`.
At most two new processes and four total GPU processes; do not signal other jobs.
No new GPU smoke: reuse completed original8token/4round smoke1886.161MiB.
Keep eight BEST weights and exact predictions per seed plus recovery state.
Automatic final summary includes all three seeds,per-rate/high and sample SD.
No cross-seed positional common-mask replay,additional Flat runs or HPO.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  /path/to/controller/experiments/osram_meld_b3_default43_3seed_20261016/run.py \
  --controller-sha SOURCE_COMMIT
```

Output:`/home/qinjinghui/remote_experiments/osram_meld_b3_default43_3seed_20261016`.
The controller verifies all original129source hashes,feature SHA,original smoke,
source weights,resource limits and duplicate/resume queue before scheduling.
Existing original per-rate Test-best checkpoints are model selections,not one
deployment checkpoint. A selected seed66does not establish multi-seed benefit.
