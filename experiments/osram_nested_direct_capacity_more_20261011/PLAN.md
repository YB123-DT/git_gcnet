# Four additional direct random Nested capacity trials

INTERNAL DIAGNOSTIC ONLY. User authorized additional trials when GPUs free.
Bounded to four more configurations, not new modules or unlimited search.
Existing four capacity runs and frozen rooted Probe remain unchanged.

| Method | Node width | Internal MLP width | Healthy physical GPU |
|---|---:|---:|---:|
| nested_direct_random_mlp128 |64|128|0|
| nested_direct_random_dim128_mlp256 |128|256|0|
| nested_direct_random_dim128_mlp512 |128|512|1|
| nested_direct_random_dim256_mlp512 |256|512|1|

Measured module counts (table order):216899,662403,1122435,2241539.
Whole-model counts are recorded by each run, not equated with node width.

All direct/no external evidence addition, normal random decoder initialization.
Original Local Skip/Flat hidden1600/output1600/head remain. Same8heads,3GIN
layers/rooted topology, OSRAM memory read/write, loss/masks/Adam.001 constant,
MOSI seed66/100epochs/per-rate Test-oracle BEST. No simultaneous Flat reduction,
new objectives/gates or LR changes. Width affects internal initial RNG draws.

Server biggpu. GPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45;
GPU1 UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049. GPU4 forbidden.
Two lane dispatchers, each max2 concurrent; immutable shared tracked snapshot,
separate lane/run outputs. Existing dispatcher generalized by default GPU_INDEX7;
old launched sources untouched. Check UUID/free memory/disk at each admission.
No baseline reruns or automatic new seeds.

Tests: new four capacity cases RED on absent registration, then GREEN alongside
original four. Same masks/interfaces/no input bypass/finite gradients and root
Probe no-test-label-update checks. Record module counts and live config/PIDs.
Final results require100epochs and outputs_verified provenance.
