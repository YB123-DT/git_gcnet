# GPU4 one-off diagnostic — INCOMPLETE

Explicit user authorization covered a short diagnostic only, not training or removal of the GPU4 ban.

- Host: `biggpu`; physical GPU4, `GPU-46fb379f-cc90-dc82-9b5e-5d011f552264` (V100 32GB).
- Environment: `/data2/yb/reproduction_workspace/envs/s0/bin/python`, torch 2.2.2+cu121.
- Remote artifacts: `/data2/yb/remote_experiments/gpu4_health_20261004/`.
- Started 2026-10-04 10:30:23 UTC; configured timeout 120 seconds with 10-second kill grace.
- No GPU reset, driver change, foreign-process termination or training queue admission.

## Observed results

1. CUDA initialization passed.
2. FP64 matrix product versus CPU reference passed (max absolute error 9.24e-14).
3. FP32 matrix product versus CPU reference passed (max absolute error 1.76e-5).
4. 8 GiB allocated memory passed four uniform byte patterns: 0, 85, 170, 255. This is not exhaustive VRAM testing.
5. CPU/GPU backward gradients passed; both losses 126.61585235595703.
6. During the subsequent intended 20-second Adam exercise, the timeout-wrapped process was killed: shell exit 137, SIGKILL. No final `passed` event was produced.

The entire operation ended before the configured 120-second timeout. Cause of SIGKILL is **unknown**; do not call this a hardware failure, a timeout, or a successful stress test. No automatic retry was performed.

## Read-only follow-up

At 10:31:06 UTC the card was idle again, 35 C, used VRAM 0 MiB, uncorrected volatile/aggregate ECC 0/0. Parent user cgroups reported `oom=0`, `oom_kill=0`, `oom_group_kill=0`, with no memory.max limit. Host available RAM was about 428 GiB. Kernel logs were inaccessible (`dmesg: Operation not permitted`); these checks cannot identify the sender of SIGKILL or certify the hardware.

Conclusion: basic computation and limited memory checks passed, but sustained execution was not verified. Keep GPU4 excluded from experiments. Investigate the termination with server administration before treating this card as a dependable training resource.

## Invocation

```bash
CUDA_VISIBLE_DEVICES=GPU-46fb379f-cc90-dc82-9b5e-5d011f552264 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout -k 10s 120s /data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 /data2/yb/remote_experiments/gpu4_health_20261004/selfcheck.py
```
