# MOSI68 cyclic stage2: launch evidence

Status: **running**, no final score claim. INTERNAL TEST-ORACLE SCREEN.

- Source: `4d5aa09d999b49d7194f72f6bf75c8ab75edbef1`.
- Server/env: biggpu,cmerc-repro Torch1.13.1+cu117.
- Snapshot: `/data2/yb/remote_experiments/osram_signature_cyclic68_20261025/source_4d5aa09`.
- Output: `/data2/yb/remote_experiments/osram_signature_cyclic68_20261025/execution_v1`.
- tmux: `signature_cyclic68_20261025`; controller PID4002119.
- Eight independent MOSI68 runs freeze the SAME corresponding per-rate Flat
  parents as the completed fixed-rate control. Only stage2 TRAIN rate scheduling
  changes to original cyclic0–0.7. Each run still TESTs/selects only its own rate.
- Unchanged SignatureBlock,batch32,Adam WD1e-5,150epochs,LR3e-4 then3e-4/10
  after100. Original MSE includinggold0; no extra losses or trained Flat.

15 remote unit/module tests passed. One actual full-batch two-update smoke
passed with zero parent-prediction replay error, frozen-parent integrity,
finite/nonzero branch gradients and exact selected correction replay. Measured
allocated smoke peak377.08MiB. Formal8processes were observed running:

| Physical GPU | Rates | PIDs |
|---|---|---|
|2|0.0–0.3|4002701,4002702,4002703,4002704|
|3|0.4–0.7|4002705,4002709,4002710,4002711|

GPU2 UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`;
GPU3 UUID `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`. GPU4 excluded;
the ongoing MOSEI jobs on GPU0/1 were not altered.

After all150epochs/300updates finish,the controller compares Flat/fixed/cyclic
per rate and mean/high. It verifies actual consumed rate sequence/counts and
identical training mask traces across the8parents. The inherited finite
schedule has counts37,38,38,38,38,38,37,36,not perfectly equal counts; no sampler
change is made to force equality. Historical fixed runs lack a separately saved
initial Signature digest; same seed/source/config convention is recorded
without inventing retrospective initialization evidence.
