# GPU1 migration preflight: not executed

User corrected the requested destination fromGPU2 toGPU1. No GPU2 task was
signalled or changed. TargetGPU1(UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049)
is healthy,ECC0,about32GiB free. The original five jobs remain onGPU7.

## Why strict full-state migration was not performed

All five `last.pt` files contain model,72Adam states,CPU/Python/NumPy random
state,history,identity and BEST ledger,but CUDA RNG lists are empty. Observed
saved epochs at the check:MOSI67/68=35/35;MOSEI66/67/68=5/4/4. These are
live-progress snapshots,not final performance results.

A separate tiny GPU1 probe reproduced the cmerc Torch1.13.1+cu117 behavior with
UUID CUDA_VISIBLE_DEVICES binding:

```text
torch.cuda.is_initialized() after allocation: True
torch.cuda.device_count(): 0
torch._C._cuda_getDeviceCount(): 1
len(torch.cuda.default_generators): 1
len(torch.cuda.get_rng_state_all()): 0
len(torch.cuda.get_rng_state(0)): 816
```

The existing checkpoint helper calls `get_rng_state_all()`,which therefore
silently records no CUDA generator state in this environment. A valid model
and optimizer checkpoint is not sufficient to promise an exact stochastic
continuation. Explicit per-device RNG capture would address future checkpoints,
but cannot reconstruct the missing state of an already saved checkpoint.
No training source or RNG handling was silently patched in these live runs.

## Actions and preserved state

The command-verified controller and five exact worker PIDs were briefly stopped
withSIGSTOP to inspect a stable checkpoint set. A guard rejected missing CUDA
RNG state before any original process was terminated. The exception path sent
SIGCONT to all paused processes. Subsequent advancing epoch summaries confirm
the original GPU7 training continued. No duplicate/replacement jobs were
launched,no weights were loaded into a new run,and no checkpoint was deleted.

Remote operational receipt:
`/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/MIGRATION_gpu1_20261023.json`.
Its status is `migration_failed`,`terminated_original_processes=false`.
The first draft expected one CUDA RNG state and failed safely;its intermediate
checkpoint-transaction wait did not repair or alter the saved RNG fields.

## User decision needed

Either leave the existing five jobs uninterrupted onGPU7,or explicitly accept
moving model+Adam+CPU RNG/epoch toGPU1 while resetting the missing CUDA random
state,recorded as non-exact stochastic continuation. Do not call the latter an
exact full-RNG resume. Do not rerun fromBEST or from scratch without approval.
The four user-cancelled conditional second-read jobs remain stopped.
