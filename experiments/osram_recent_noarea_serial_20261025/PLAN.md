# Recent NoArea serial interaction implementation plan

> Execute with scoped native subagents under AGENTS.md and tests-first; the user
> approved the structure and explicitly requested BOTH training stages. Latest
> correction: NoArea (S1 plus symmetric S2), NOT first-order-only.

Goal: MOSI66 parallel versus serial NoArea recent-history processing, stage1 and
stage2 separately. No new seeds,other datasets,HPO,loss,Gate,Memory rewrite or
mask protocol changes. Original artifacts remain protected.

## Locked architecture and interfaces

Original path: Encoder/CHM -> Local256,Base1024,Gap3x1024 -> Flat4352->1600 plus
original LocalSkip -> original LayerNorm/head. Only forward512 memory dimensions
contain causal evidence; backward512 remain zero. Read-before-write,write_step.6,
joint ATV BlockWrite and all queries remain unchanged.

Parallel: reuse exact NoAreaSignatureBlock (281856parameters),window8. Its532
descriptor contains three90d signatures (S1=9,symmetricS2=81),Local256,
availability3,counts3. Zero-initialized1600exit adds to original pre-LN anchor.

Serial: same three projected9d paths and symmetricS2,NO antisymmetric area.
History vector = signature270 + counts3 (273),not currentLocal/availability.
History273->64 (bias-free Linear+GELU),Local256->64,shared read512->64,
four role embeddings8. Shared D: concatenate[local64,read64,history64,
availability3,role8]=203 -> LayerNorm203 -> Linear128 -> GELU -> Linear64.
For each role r,delta = zero-initialized bias-free Decoder_r64->512 applied
to D(L,R,p,a,r)-D(L,R,0,a,r). No stochastic layers in these two evaluations.
Base and currently missing Gap roles update as R+delta;inactive Gap,padding and
no recent observed history have zero correction. Local/LocalSkip unchanged.
Updated reads enter the ORIGINAL Flat,Norm and head. No global1600serial exit.
Counts/descriptor are strict-past,real observed only;no completion/futurelabels.
Different parallel/serial parameter counts are reported,not called equal-capacity.

Model API: RecentNoAreaModel(parent,mode='parallel'|'serial',stage='joint'|'frozen').
forward(view,generator=None,query_keys=None) returns prediction,base_prediction,
aux_loss=0. One live Encoder/CHM forward using temporary hooks only; hooks
removed in finally. Serial hook is before emotion_adapter,parallel before
emotion_norm. Do not globally edit model.py/osram.py or existing experiments.
Frozen train() leaves parent.eval()/requires_gradFalse but retains input-gradient
through frozen Flat/Norm/head to new serial reads. Joint all parent parameters
remain trainable. predict_frame(frame) supports frozen cached original FlatFrame
evaluation without rescanning CHM. Training need not compute branch-off output;
evaluation may decode branch-off with no extra scan and no training RNG effect.

## Training protocol

Numerical source e0ca94c, cmerc-repro,biggpu,seed66,officialsplit,batch32,Adam,
WD1e-5,MSE includinggold0,original cyclic0..0.7 GCNet-random-mask-v1,
150epochs,LR3e-4 for100 then ORIGINAL 3e-4/10 expression for50. Test-best WF1
earliest strict tie,ACC from same predictions;exact protected Flat test masks.
No auxiliary objective. New module construction is RNG-isolated;fresh joint
parent must match accepted initial_backbone SHA. Record complete source/config,
data/environment,mask/traintrace,params,optimizer/RNG states and selected arrays.

Stage1:2fresh joint models,parallel/serial,each tests all8rates every epoch.
Compare accepted same-protocol Flat66 only after source/init/config/mask parity
checks. Original13.509793M Flat,not old80.7/missing variants.
Stage2:16branches,parallel/serial x8rates. Each loads own protected Flat BEST,
freezes all original parameters,trains new block150epochs with cyclic training,
tests only designated exact rate. Reuse completed parallel NoArea66 only if
complete config/init/LR expression/trace identical; otherwise run paired new
parallel explicitly. Stage1/2 score tables remain separate;extra training budget
and per-rate multiple-parent models are disclosed. Internal Test-oracle only.

## File ownership and steps

- [x] blocks.py,model.py,tests/test_recent_noarea_serial_model.py: write failing
  tests,implement minimal module/wiring,verify no direction,causal/window/masks,
  initial identity,zero-history centering,backward zeros,gradients,freeze/train,
  one scan,hookcleanup and state restore. Own only these files.
- [x] stage1.py,tests/test_recent_noarea_serial_stage1.py: reuse original joint
  setup/train/evaluate helpers,lock finite two-mode scope,complete RNG resume,
  selected artifacts,initial identity/source/mask parity,finite gradients and
  actual updates. Real smoke3updates;do not launch. Own only these files.
- [x] stage2.py,tests/test_recent_noarea_serial_stage2.py: reuse pinned prepare
  and original cyclictrain helpers;own-rateparent/masks,originalfrozenidentity,
  cached predict_frame evaluation,complete RNG resume. Verify actual parent
  no-grad/digest and branch gradients,selected replay. No launches. Own only
  these files.
- [x] scheduler.py,summarize.py,README/tests: root integrates modules,validates
  all focused tests,deploys committed immutable overlay and preserves sourcee0.
  Four real-data GPU smokes (stage xmode) gate18formaljobs unless a strictly
  reusable parallel stage2 result is found. UUID physical0/1/2 only,max6/card,
  use live VRAM/RAM/disk gate and stagger. No external processes signalled.
- [ ] Run finite automatic queue in tmux,verify live workers and produce launch
  receipt. Completion independently recomputes saved ACC/WF1 and selected hashes;
  final RESULT.md/JSON/CSV includes all8rates/high,stages separate,failures and
  parameters/cost. Commit/push only task code/tests/small records,no weights.

Verification commands:
`python -m pytest -q tests/test_recent_noarea_serial_model.py tests/test_recent_noarea_serial_stage1.py tests/test_recent_noarea_serial_stage2.py tests/test_recent_noarea_serial_queue.py`
then compileall/diff check,real GPU smoke,formal live status and final validated
selected predictions. Smoke never counts as effectiveness evidence.

Implementation checkpoint: 38focused localtests passed. Independent read-only
spec/quality review found and verified the fix for blocked-unlaunched queue
recovery;serial first-step zero-gradient behavior now has an explicit3update
regression without changing zero initialization. Real GPU smoke/deployment are
the next gate,not yet claimed complete here.
