# Paired training gradient observer launch

INTERNAL DIAGNOSTIC ONLY. Running, not completed. No performance conclusion.

MOSI seed66 only, original Flat and original Nested each100epochs from scratch
to obtain gradients throughout training. These are authorized monitored reruns,
not recovery of the previously completed100/150epoch experiments. No Gate.
Common historical source ad211c0, observer wrapper597e9a1. Tests:2 passed in1.67s,
one existing pynvml deprecation warning. Outputs/gradients/RNG parity, masks,
Local Skip and evaluation exclusion checked on small autograd graphs.

| Model | tmux session | PID |
|---|---|---:|
| Flat | gradtrain_flat_66 | 2523945 |
| Nested | gradtrain_nested_66 | 2524065 |

Server biggpu, GPU7 UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e. Pre-launch
GPU7 free17967MiB; two jobs admitted without modifying batch32. Independent
remote root /data1/yb/remote_experiments/osram_nested_training_gradients_20261010.
Sealed observer scripts in source/; historical model source remains untouched.
Logs: logs/{flat,nested}_seed66.log. Outputs: runs/{flat,nested}_seed66.
Per-epoch statistics: gradients/epoch_NNN.json within each output.

At2026-10-10T04:35:24Z both were live, provenance running and epoch1 committed.
Both generated2 step records (rates0.0/.1), with equal training availability
hashes. Local includes843 valid rows in the first batch; Base excludes first
turns and includes811. No Gap is active in that first complete-modality batch.
Native input/parameter gradients finite, with strict mask counts verified.
First-step Local gradient and common parameter gradient norms match exactly;
Nested's zero residual and initially zero Nested gradient are expected with
zero-initialized readout bridges, not evidence of later gradient vanishing.

Initial per-step examples and launch provenance are cached here; they are not
final summaries. When complete, compare paired epoch/rate gradients across full
cyclic windows, clipping exposure, residual evolution and task performance.
