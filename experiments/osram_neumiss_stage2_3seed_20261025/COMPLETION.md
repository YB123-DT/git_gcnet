# NeuMiss MOSI fixed-rate stage2: completed three-seed result

All16 new logical runs completed without failure:14 trainings reached150epochs,
and two rate0.0 cases were epoch-zero exact no-ops. Eight completedseed66 runs
are reused unchanged. RESULT.md/JSON/CSV and GROUPS.csv containall24 seed/rate
records; selected predictions,checkpoint/mask hashes,metadata order and metrics
were independently revalidated for all24 after completion.

| Seed | Parent ACC | NeuMiss ACC | Parent WF1 | NeuMiss WF1 | ΔWF1 pp |
| --- | ---: | ---: | ---: | ---: | ---: |
|66|79.535061|79.630335|79.460447|79.612215|+0.151768|
|67|79.668445|79.744665|79.479841|79.703054|+0.223212|
|68|79.173018|79.077744|79.074201|79.089659|+0.015458|
|Mean|79.458841|79.484248|79.338163|79.468309|+0.130146|

NeuMiss seed-mean sample SD: ACC0.356654,WF10.331051 (n=3). Paired WF1 gain
sampleSD0.105552. These are descriptive seed statistics,not evidence of
statistical significance. Rates are not independent seed repeats.

Mean MSE changes1.463359→1.472856 (worse by0.009497),despite a small mean WF1
increase. Seed68 ACC decreases; negative rates and metric conflicts remain in
the full results. This does not establish reliable regression improvement.

Numerical source `e496b4842e1871b6a946dc9623ff6f2221b2828e`,cmerc-repro,biggpu,
unchanged NeuMiss245984parameters,frozen original parents,batch32,Adam WD1e-5,
original GCNet-random-mask-v1,MSE includinggold0,and current-rate-only stage2.
Per-rate Test-best WF1 uses same-epoch ACC/MSE/exactmask. Extra stage2 budget and
test-based selection remain INTERNAL TEST-ORACLE exploration,not independent
generalization evidence or a single-checkpoint eight-rate deployment score.

Remote results and full checkpoints remain at
`/data2/yb/remote_experiments/osram_neumiss_stage2_3seed_20261025/execution_v1`.
No new experiment or automatic Signature resume was triggered by completion.
The user's later explicit Signature continuation is separately documented in
`experiments/osram_signature_iemocap_cyclic_20261025/RESUME_GPU012_20261025.md`.
