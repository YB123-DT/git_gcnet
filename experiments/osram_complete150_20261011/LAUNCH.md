# Verified persistent launch

INTERNAL DIAGNOSTIC ONLY.62authorized tasks; none completed at this snapshot.

Server biggpu. Root:
`/home/qinjinghui/yb_gcnet_experiments/osram_complete150_20261011`.
Same-host root filesystem had492GiB available before launch. Sealed wrapper
`source_4eed138/run.py`, commit4eed138. Historical model/trainer sources unchanged:
ad211c0(MOSI),33ded09/644ee14(IEMOCAP, matching original provenance).
MANIFEST.json contains62tasks, all source/config hashes, data hashes and checked
metadata;30original Nested recovery hashes validated. Original files untouched.

Persistent tmux socket `gcnet_complete150_20261011`, session `complete150`.
Coordinator PID2149106. Python `/data2/yb/reproduction_workspace/envs/s0/bin/python`.

```sh
python -u source_4eed138/run.py --root /home/qinjinghui/yb_gcnet_experiments/osram_complete150_20261011 --dispatch
```

State `QUEUE.json`, dispatch log `dispatcher.log`, per-task logs `logs/`, outputs
`runs/TASK/stage100` (fresh Flat only) and `runs/TASK/stage150`.
Queue automatically fills authorized slots, halts new admissions on failure.
Whitelist0/1/2/3/7, max2jobs/GPU; badGPU4 excluded. GPU0 became occupied by other
work between initial scan and launch, so scheduler skipped it without signals.

Initial verified snapshot:8running,54pending,0failed,0complete.

|Task|GPU|PID|Observed progress|
|---|---:|---:|---|
|MOSI Flat67|1|2149132|5completed epochs|
|MOSI Flat68|2|2149195|4completed epochs|
|IEMOCAPFour Flat66 fold1|3|2149731|1completed epoch|
|IEMOCAPFour Nested66 fold1|7|2150317|101completed epochs; all group LR1e-4|
|IEMOCAPSix Flat66 fold1|1|2151503|data loading|
|IEMOCAPSix Nested66 fold1|2|2151616|initialization|
|IEMOCAPFour Flat67 fold1|3|2152153|data loaded|
|IEMOCAPFour Nested67 fold1|7|2152639|initialization|

Verified targeted local tests:62unique tasks,2MOSI/30Nested/30IEMOCAPFlat,
only epoch/LR changes for continuation, optimizer-state-preserving LR override,
GPU4exclusion, Python compile. First actual Nested continuation passed strict
model/optimizer/RNG/schedule recovery and produced epoch101 history. First Flat
runs produced committed full recovery states. No performance/completion claim
from early epochs. Later progress must be read from queue and artifacts.

No rho0.5paired-view task launched: user explicitly prioritized items1/2.
