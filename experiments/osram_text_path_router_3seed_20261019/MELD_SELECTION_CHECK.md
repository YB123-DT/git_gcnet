# MELD selection and preservation check,2026-10-20

Read-only audit; no training, inference, checkpoint deletion or new jobs.

The protected originalFlat baseline remains per-rateTest-best ACC:
ACC61.55651340996169,UA37.727516415083656,W-F1 58.50876476707808.
Source:`../osram_meld_wd_seed66_20261018/BASELINE.json` and actualremote
metrics/configs in S66_WD1,S67_WD1,S68_WD1. Allthree remote metrics explicitly
record per-rate-test-oracle/accuracy,GCNet-random-mask-v1,50epochs. All24BEST
weights and three last_training.pt files still exist. This inspection checked
existence and metadata; it did not rehash allbaselineweights or re-infer them.

The later sixrun pairedFlat/Router round used validation-eight-rate ACC then
lowerCE,ONEselectedcheckpoint perrun. It retained all50epoch training/validation
rows but ZEROrows with test scores; only best.pt and last.pt modelstates remain.
Testing happened after validation selection. Those scores cannot be relabelled
as historical per-rateTest-best using only the saved history.

| Round | Selection | ACC | UA | W-F1 |
|---|---|---:|---:|---:|
|Protected originalFlat|per-rateTest-best ACC|61.556513|37.727516|58.508765|
|Later pairedFlat|single validation-selected|60.512452|38.044884|57.783426|
|Later pairedRouter|single validation-selected|60.478927|38.034694|57.744009|

Allvalues three-seed equal eight-rate means. OriginalFlat versus this later
validation-selected pair is NOT a matched selection/mask comparison. The later
MELD split is its original officialtrain/dev/test,not IEMOCAP's extra10%holdout.
Within the later matchedvalidation-selected pair,Router minusFlat is-0.033525pp
ACC,-0.010190ppUA,-0.039417ppW-F1. Do not claim this is the requested MELD
Test-bestRouter result. No such correctedMELDRouter rerun is launched here.

Remote originalpairseed66:
`/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/runs/MELD/{flat,joint}`.
Seed67/68:
`/home/qinjinghui/remote_experiments/osram_text_path_router_3seed_20261019/execution_v1/runs/MELD/seed_{67,68}/{flat,joint}`.
OriginalFlat need not be retrained merely to redisplay its savedbaseline score;
any future strictpaired rerun must explicitly state its selection and masks.
