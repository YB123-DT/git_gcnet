# Nested-only lowerLR/warmup, random residual decoders

INTERNAL DIAGNOSTIC ONLY. One MOSI seed66,100epochs, from scratch; no scores yet.
Code4825fee pushed before deployment. Seven regression tests passed in2.22s;
pre-existing pynvml deprecation warning only. Isolated sealed-source runtime
policy; original default code and any existing experiment source untouched.

Server biggpu physicalGPU7, UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e.
Root /data1/yb/remote_experiments/osram_nested_local_lr_warmup_random_20261010.
Prelaunch GPU7free19685MiB; /data1free76GiB. No unrelated process terminated.
PID3321622; tmux socket gcnet_nested_lrw, session nested_lrw_seed66.
Initial provenance running and PID alive. First epoch completed. Actual epoch1:
original backbone/projector/classifier LR1e-3, independent Nested LR1e-4.
Partition parameter counts12578912/929280/1601/159235, no omitted/duplicate
parameters in tested split. All9 decoder weight norms nonzero: Local9.263942,
eight Memory decoders4.56–4.69. Random branch is active at initialization;
initial equivalence to Flat is intentionally lost. First train task loss99.5079
is large but finite; it is not a performance benefit or a reason to claim success.

Original source_ad211c0 TokenAdapter predates zero_decoder argument. During
adapter construction only, its zero_linear factory is replaced with nn.Linear
and then restored in finally. Keeps the exact original residual forward and RNG
draw order; no direct-evidence variant or extra decoder reinitialization.

```sh
env CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u /data1/yb/remote_experiments/osram_nested_local_lr_warmup_random_20261010/source/run.py --source /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0 --reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --output /data1/yb/remote_experiments/osram_nested_local_lr_warmup_random_20261010/runs/nested_seed66 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --wrapper-commit 4825fee --nested-lr .0005 --warmup-epochs 5 > /data1/yb/remote_experiments/osram_nested_local_lr_warmup_random_20261010/logs/nested_seed66.log 2>&1
```

Nested module includes tokenizer/core/decoders. Target5e-4, epochs1–5linear
warmup1/2/3/4/5e-4, then constant5e-4. All other groups1e-3 throughout.
Original global gradient clipping1.0 unchanged. Stage-specific checks and final
scores recorded separately; job submission alone is not evidence of completion.
