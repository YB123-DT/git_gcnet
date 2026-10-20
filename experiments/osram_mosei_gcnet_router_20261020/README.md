# MOSEI addition: Flat versus one-stage Text-source Router

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

User2026-10-20 requested addingMOSEI to the current GCNet-missing experiment.
First finite screen:seed66 only,two fresh runs,Flat and one-stageRouter.
No automatic extra seeds,methods,losses or hyperparameter search.

## Configuration and provenance

- Use the actual protected MOSEIFlat config from
  `/data2/yb/remote_experiments/osram_mosei_cfg84_nojepa_20260919/seed_66/config.json`,
  SHA256`2b612c7093c696121f6e92762b24a75acf6db316106f2ff07b240603df529155`.
  Read configuration only; never initialize from its trained weights.
- Standing MOSEI budget60epochs, as documented in
  `../osram_mosei_dim32_depth2_followup_20261012/LAUNCH.md` and earlier user
  instructions. Do not copy MOSI150epochs or its100+50LR schedule.
- OriginalFlat1600/LocalSkip/causalCHM,write_step0.6,8heads,key/value64,
  latent256,emotion-only/noJEPA,regressionMSE,dropout0.5,batch32,
  Adam/backboneLR1e-3,wd1e-5,constant schedule. RouterLR3e-4,constant.
- Only the same accepted one-stage Text-sourceRouter differs between arms.
  No Nested,TASRAdapter,completion,persistentmix,pairedviews or newloss.
- `gcnet-random-mask-v1`,originalhost/guest paddedrectangle sampling,
  cyclic rates0.0–0.7. At0.7 every valid utterance observes one modality;
  padding zero. Current missing input never becomes synthetic observed input.
- Use original official MOSEI train/dev/test; no extra training holdout.
  Evaluate eachrate everyepoch; independently select testW-F1BEST,strict
  improvement,earliesttie. ACC/MSE/predictions/masks from the same call.
- Fixed legal GCNet testmaskbank common across the two arms,isolatedRNG;
  matching fresh initialization,data order and actualtrainingmask hashes.
  Old conversation-mask scores are not same-mask paired baselines.

Data manifest:
`/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json`,
SHA256`fd03c4a6ca12abe8a55bde88531b7103761c4ed263c9fe381e30809aabd2a46c`.
The MOSEI entry pins68,569feature/label files,original roots and hashes.
Feature encoders: wav2vec-large-c-UTT,deberta-large-4-UTT,manet_UTT.
No new extraction or feature substitution.

Runtime is the existing MOSEI s0 environment:
`/data2/yb/reproduction_workspace/envs/s0/bin/python`,Torch2.2.2+cu121,
NumPy1.24.3,sklearn1.3.2. Its use is recorded in prior MOSEI LAUNCH.json.
The shared scalar helper now explicitly accepts CMUMOSEI and passes its real
dataset to task loss/metrics. Default MOSI outputs remain regression-tested;
running MOSI/IEMOCAP source snapshots are not edited or restarted.

## Persistence and execution

Everyrun keeps full config/source/data/environment/split/seed/GPU provenance,
all60epoch eight-rate scores,eightBEST with exact selected predictions/masks,
full `last.pt` with Adam/RNG/history and selectedledger. Publishing reuses
selected arrays,not another random inference. Final report checks exact masks,
paired training traces and same-call recomputed metrics.

One real3batchRouter GPU smoke gates formal training; verifies finite gradients,
actual updates,full recovery and exact selectedprediction replay. Smoke is not
a performance result. Queue is persistent via tmux and resource-admitted:
one healthy GPU,max2children,RAM>=64GiB,measuredVRAMmargin,disk>=16GiB.
GPU4forbidden;no other tasks killed/moved or batchsize reduced for capacity.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosei_gcnet_router_20261020.run \
  --output /data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
```

Single-seed,exploratoryper-rateTest-oracle results are not independent test
generalization. Mean8 may combine eightcheckpoint epochs,not one deployedmodel.
