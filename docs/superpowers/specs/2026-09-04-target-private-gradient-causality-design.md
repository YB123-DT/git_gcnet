# Target-Private Gradient Causality A/B Design

## Question

Does forcing nearly orthogonal missing-target tasks through the same Missing-M3
MMoE experts reduce downstream MOSI performance?

This experiment tests harm from forced sharing. It does not claim that
orthogonality is intrinsically harmful and does not use PCGrad, which would test
negative gradients rather than near-zero gradients.

## Locked intervention

- Control: current Missing-M3 with `target_private_rank=0`.
- Treatment: the identical model with `target_private_rank=32`.
- The treatment adds one zero-initialized low-rank residual per target modality.
  Shared experts, gates, heads, GCNet, losses, masks, data and optimizer remain
  unchanged.
- Target-private parameters receive gradients only when their corresponding
  target modality is missing.

## Protocol

- Dataset: CMU-MOSI.
- Seeds: 66, 67, 68, 69, 70.
- Training: one stratified mixed-rate model per arm and seed.
- Evaluation: missing rates 0.0 through 0.7 from the same checkpoint.
- Selection: validation weighted F1 only; test is evaluated after selection.
- Pairing: same seed, split, deterministic mask schedule, initialization seed,
  epoch budget and command options across arms.
- GPU 4 is excluded. Completed compatible jobs are inherited atomically.

## Measurements and decision

Primary measurement is per-seed treatment-minus-control test weighted-F1 at
each rate. Also report mean non-zero-rate delta, positive-seed count, parameter
delta and collapse checks.

Evidence that forced sharing is harmful requires:

1. positive mean delta at a majority of non-zero rates;
2. at least 3/5 positive paired seeds for the non-zero-rate macro average;
3. no material miss-0 degradation or class-output collapse.

If these conditions fail, near-orthogonal target gradients are treated as task
independence rather than a demonstrated performance problem. If they pass, a
later parameter-matched target-independent residual is required before claiming
that target privacy, rather than added capacity, explains the gain.

## Artifacts

The runner writes one isolated directory per arm and seed, plus an atomic
manifest and paired summary. No Original GCNet job is part of this experiment.

