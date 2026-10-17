# MELD alternate released features, original Flat seed66

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC. Only data changes; no Block2,
Nested,Gate,JEPA,completion,extra loss or optimizer search. Formal jobs are
biggpu-only in `/data2/yb/conda_envs/cmerc-repro/bin/python`,not local training.

Fixed:seed66,50epochs,batch32,Adam LR0.001,WD0.00001,dropout0.5,
Flat hidden/output1600,original Local Skip,CHM8heads/64K/V,write0.6,causal
read-before-write/joint block write,cyclic0.0–0.7,`gcnet-random-mask-v1`,
per-rate Test-best ACC. Same-selected-epoch ACC/7class UA/W-F1/CE and per-class
recalls are preserved with every BEST weight and exact mask/prediction artifact.
Numerical source is immutable `737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`.

## Feature releases and boundaries

| Release | Verified A/T/V dimensions | Availability |
|---|---|---|
| MAGTKD first stage |768/768/768|all9989train/1109dev/2610test available and converted|
| MM-Align |32/300/2048 after fixed temporal mean|train download incomplete; dev missing110_7|

MAGTKD source:[official repo](https://github.com/JieLi-dd/MAGTKD),commit
`95d0760c26ad2a6ad2181daf372abfd5be348b2d`;[download](https://drive.google.com/file/d/19g3hTaBEKF5wXI0DHdvRYbu0BD3XZa3d/view).
Use `text/audio_kd/video_kd`,as its original fusion code does. These features
already include upstream task supervision and cross-modal distillation;
text uses a causal dialogue prefix. No new distillation is trained here.
Do not present this as a controlled swap between generic frozen encoders.

MM-Align source:[official repo](https://github.com/declare-lab/MM-Align),commit
`e2448590b3d9b308350515bcb147526f4baa6ac7`;[download](https://drive.google.com/file/d/1RjrYSMpXxg_6r_nUQaysaPyMsldLpMcb/view).
Actual aligned audio is32dim,not an assumed768dim wav2vec hidden state.
Lookup the released6336x300embedding using original token IDs and mean-pool
each aligned sequence without learning a new encoder. Release dictionary order
is label-grouped,so reconstruct chronological conversation/utterance order.
Do not drop or fabricate missing source utterances to force a training run.

`prepare.py` preserves canonical fields0/1/2/13/14/15/16 exactly,including all
speaker selectors and official split ordering. Only feature arrays change.
MAGTKD speaker indices use a different global mapping and never replace these
selectors. Its upstream train556_6speaker differs (Ross versus canonicalJoey);
the canonical selector is retained and the source discrepancy recorded.
MAGTKD's CSV also has text preprocessing changes; canonical joining uses official
MELD CSV IDs,labels,text and speakers,not an assumption of identical raw strings.

## Preparation and training

```bash
python -m experiments.osram_meld_alternate_features_20261017.prepare \
  --release magtkd \
  --source /data2/yb/paper/feature_sources/magtkd_20261017 \
  --canonical /data2/yb/paper/baseline_sources_20261006/ECERC/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --metadata /data2/yb/paper/feature_sources/mmalign_20261017 \
  --output /data2/yb/paper/feature_sources/converted_20261017/magtkd_meld.pkl
```

Prepared SHA256:`59eb11b079fc02eea2a38f81bc1132ede29077810697d3d9384c1796450ecb13`.
The existing data loader accepts this17field bundle directly. Input projector
parameter shapes follow the new dimensions; Flat/CHM architecture does not change.
Input dimension changes can change initialized parameter counts/RNG consumption;
do not promise byte-identical starting weights or matched winning-epoch masks.

`worker.py` verifies131source hashes,exact Flat config,data hash,environment data
binding and GPU UUID. Optional three-batch GPU smoke is separate from formal
training and does not supply a performance result. Formal training preserves
full optimizer/RNG recovery and uses no additional stochastic inference to fill UA.

`submit.py` is a bounded local connection/submission helper,not local training.
It retries only the authorized MAGTKD job on `ssh biggpu` at60second intervals,
up to1440attempts. On connection recovery it verifies the same Torch/NumPy
environment,chooses healthy physicalGPU3/2/0 with at least6GiB free and fewer
than4total processes,uploads hash-pinned data/code,and starts one persistent
smoke-then-train tmux chain. It never installs dependencies or signals others.
Worker code is stored remotely under its content hash. MM-Align is NOT submitted
while source coverage is incomplete. Do not restart submission without checking
`SUBMISSION.json`,the remote tmux session,and worker SUMMARY files.

Remote target:
`/home/qinjinghui/remote_experiments/osram_meld_alternate_features_20261017`.
Local submission state:
`/data2/yb/paper/feature_sources/alternate_flat_submission_20261017`.

All new feature files remain outside Git. Only code,source/data hashes and small
audit summaries are committed. Existing ECERC baselines are not replaced.
