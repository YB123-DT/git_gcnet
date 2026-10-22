# Capacity allocation implementation checks

INTERNAL TEST-ORACLE SCREEN. Implementation checks are not performance evidence.

Immutable training source: `5a86f7e98bcbe8eead96b10410be6175d0abd832`.

## CPU regression checks

44 tests passed (8.29 seconds), covering:

- Old V3 default state keys/shapes, initialization, RNG, output and gradients.
- Config/CLI propagation and rejection of invalid widths.
- Wider V3 shapes, inactive evidence, no-history/padding and finite gradients.
- Zero-init equivalence to the same smaller Flat, not the original larger Flat.
- No unrelated original parameter initialization changes.
- Exactly two finite seed66 MOSEI arms; no missing-protocol or objective changes.

```bash
python -m pytest -q \
  tests/test_mosei_v3_capacity_runner.py \
  tests/test_source_role_v3_capacity.py \
  tests/test_source_role_v3.py \
  tests/test_source_role_v3_integration.py \
  tests/test_source_role_v3_runner.py
```

The local `multimodalerc310` interpreter was used with the installed pytest path
added to `sys.path`; the remote real-data checks use archived MOSEI `s0` instead.
`py_compile` for worker/run/report and `git diff --check` also passed.

## Remote execution

Server: `biggpu`; physical GPU1:
`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
GPU4 is excluded. At admission GPU1 was empty with 32,491 MiB free.
RAM available was about 415 GiB; `/data2` free space about 250 GiB.

Persistent controller started in tmux socket `osram_capacity`, session
`mosei_capacity`, PID2650147. Source and output:

```text
/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/source_5a86f7e
/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1
```

The controller runs two three-batch real MOSEI forward/backward/Adam smokes,
including optimizer/RNG/prediction reload, before admitting formal training.
Actual smoke and launch outcomes are recorded separately in `LAUNCH.json`.
Do not infer training completion from the controller startup alone.

No existing training process, result directory or checkpoint was modified.
