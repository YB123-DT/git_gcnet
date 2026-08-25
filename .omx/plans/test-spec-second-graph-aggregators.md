# Test Specification: GCNet Second-Layer Aggregators

Chinese mirror: [test-spec-second-graph-aggregators.zh.md](test-spec-second-graph-aggregators.zh.md). English mirror: [test-spec-second-graph-aggregators.en.md](test-spec-second-graph-aggregators.en.md).

## Unit proof

1. Legacy add: compare two identically seeded GraphModels constructed with implicit legacy defaults and explicit `second_graph_aggregation="add"`; require identical CPU RNG state, parameter keys/count/tensors, outputs, and gradients.
2. GenAgg formula: compare a small indexed neighborhood with a hand implementation; check empty-target output is zero.
3. GenAgg sum special case: inject identity forward/inverse with `alpha=1`, `beta=0`; require exact scatter-add output. Learned BatchNorm formula comparisons run in evaluation mode.
4. GenAgg backward: require finite gradients for inputs, alpha, beta, every forward/inverse MLP parameter, and both GraphConv linear paths.
5. GenAgg parameter count: require 59 extra trainable parameters per branch and 118 per GraphModel.
6. Soft Medoid formula: compare a three-message neighborhood against hand-computed pairwise distances and weights; require packed implementation to match ragged neighborhoods under edge permutation.
7. Soft Medoid invariants: require exact add for one neighbor and homogeneous neighbors, exact root-plus-bias for zero degree, finite heterogeneous backward, and zero parameter delta versus GraphConv.
8. Integration: require both `conv2` modules to match the selector and both `conv1` modules to remain PyG RGCN.
9. Training loss: require inverse loss only for training GenAgg, zero for add/Soft Medoid/evaluation, and gradient arrival at GenAgg parameters.
10. CLI/archive: require Python 3.8 parsing, accepted selector choices, legacy filename identity for add, and collision-free tags for candidates.

## Runner proof

1. Build exactly 12 unique jobs for two arms, two rates, and three seeds.
2. Candidate commands differ from Original only by second aggregation and output identity.
3. Parallel-arm mode makes one `run_jobs` call over all 12 `stage=formal` jobs; legacy mode remains sequential. A test must prove `stage=gate` artifacts cannot silently satisfy formal resume.
4. Resume accepts only complete immutable artifacts and never launches Original.
5. Gate rejects provenance/mask mismatch, nonfinite values, a nonpositive rate mean, a nonpositive seed macro, or fewer than two positive seeds. Historical Original archives may omit only fields introduced after their source run, which map to locked legacy defaults; any other drift is rejected.

## Runtime proof

1. Run the focused tests, then the complete local suite.
2. Synchronize a clean committed source snapshot to a real Git clone on biggpu.
3. Run exactly one FP32 candidate forward/backward per module in the official training interpreter.
4. Compare source manifests before any job launch.
5. Run the 12 tasks and validate exactly one NPZ plus 100 epoch records per task.

## Completion proof

- Full suite reports zero failures.
- Static compilation reports zero errors.
- Architect review approves scope and attribution.
- Gate report is bilingual and contains traceable task-level evidence.
