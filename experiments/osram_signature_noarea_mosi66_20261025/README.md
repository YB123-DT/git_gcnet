# MOSI seed66 matched NoArea Signature retraining

User-requested seed66 replication of the completed MOSI68 directional ablation.
Only8new branches,one per rate0.0–0.7. Reuse completed Flat and full Signature;
do not retrain either,addseed67,search hyperparameters or change missing masks.

Use the SAME NoAreaSignatureBlock as IEMOCAP68/MOSI68: keep S1 and symmetric
degree-two part0.5*S1 outer S1; remove only antisymmetric directional area at
training and inference. Endpoints,time,counts,Local,availability,window8,network
dimensions,zero-initialized outlet and281856parameters remain unchanged.

Each branch freezes its own original per-rate Flat,including Encoder,CHM,
LocalSkip,Flat,Norm and task head. Train only Signature with original cyclic
GCNet-random-mask-v1 batches; evaluate only its own saved exact test masks.
150epochs,batch32,Adam,WD1e-5,original MSE including gold0,gradientclip unchanged.
Branch LR3e-4 for100epochs,then3e-5 for50,using the exact seed66 reference code.
Test-best WF1,earliest strict tie;ACC uses the same selected epoch/prediction.
Internal per-rate Test-oracle exploration,not one deployable checkpoint.

Original numerical source is e0ca94c4c33de0efda2c2f4786fd289067e1ab2c at
`/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
Full Signature references are
`/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/execution_v2/train/CMUMOSI_66_RATE`.
The seed66 parent preparation/evaluation bridge is reused exactly; it differs
from the MOSI68 bridge and must not be substituted just because dimensions match.

New code uses a separate immutable overlay and output directory. Original source,
checkpoints,masks,predictions and previous results are never overwritten.
Verify identical parent,initialization,testmask and all150epochs of training
mask/sample/update/rate/LR traces against the completed full Signature.
Save full branch/Adam/RNG/epoch state; verify selected checkpoint replay and
frozen parent identity. Summarize eight rates and high0.5–0.7 ACC/WF1 only after
all8runs validate. Keep MSE in records but do not use it as mechanism evidence.

biggpu,cmerc-repro,physicalGPU0/1/2 via UUID only. At most3workers/card,8total,
with measured smoke VRAM/RAM and resource reserves; GPU4 forbidden,unrelated
tasks untouched. Same-RNG two-update original-vs-NoArea smoke precedes the finite
queue. tmux controller plus detached workers support explicit compatible resume.

```bash
PYTHONPATH=/path/to/new/overlay \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_signature_noarea_mosi66_20261025.scheduler \
 --root /path/to/new/execution_v1 --source-commit ACTUAL_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 3
```

This isolates directional area under this frozen-parent training budget; it
does not isolate all benefits of endpoints,time,counts or additional capacity.
NoArea matching full weakens a directional-mechanism explanation,while a full
advantage on one seed does not prove it is the universal cause of improvement.
