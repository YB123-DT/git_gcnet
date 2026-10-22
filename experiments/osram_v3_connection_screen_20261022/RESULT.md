# V3 connection screen launch

INTERNAL TEST-ORACLE SCREEN

Observed 2026-10-22T17:06:37.157284+00:00. Six seed66 runs are training on biggpu, physical GPU2/GPU7, maximum three per GPU. All six real-data CUDA smokes passed. This is a launch/status record; no final model performance or ranking is available yet.

| Run | Observed epoch | State | Physical GPU | PID |
|---|---:|---|---:|---:|
| CMUMOSI_head-preserving | 19 | running | 2 | 2373467 |
| CMUMOSEI_head-preserving | 1 | running | 7 | 2374699 |
| CMUMOSI_source-slots | 16 | running | 7 | 2375966 |
| CMUMOSEI_source-slots | 1 | running | 2 | 2377209 |
| CMUMOSI_flat-hidden | 14 | running | 7 | 2378434 |
| CMUMOSEI_flat-hidden | 1 | running | 2 | 2379615 |

MOSI: 150 epochs, first 100 at 3e-4 then 50 at 3e-5. MOSEI: 60 epochs at 1e-3. Original Flat1600, CHM8x64, batch32, Adam, WD1e-5, MSE and GCNet random missing 0.0–0.7. New variants: 1=head-preserving, 2=source-slots, 3=flat-hidden.

Runtime source: `d00d0629d3cca484526cc9718a6ade883925ca45`. Both datasets reuse the accepted Flat66 masks and check exact per-epoch training mask traces, update counts and learning rates. Each run has passed the four-update nonzero-gradient gate and produced formal epoch results. BEST checkpoints and matching predictions/masks are retained for all eight rates, alongside complete last optimizer/RNG state.

Remote output: `/data2/yb/remote_experiments/osram_v3_connection_screen_20261022/execution_v2`.
Persistent tmux socket/session: `osram_v3_connections` / `screen_v2`.
The controller automatically writes REPORT.json, PER_RATE.csv and RESULT.md as formal jobs finish; incomplete runs and smokes are excluded from score tables. Detailed source/data hashes, commands, process verification and smoke records are in LAUNCH.json and VALIDATION.md. Existing Flat scores are reused; no new Flat training was launched.

The initial code-only deployment failed before optimization because its default label directory was missing. The original dataset link was added after matching both label-file hashes; execution_v1 is retained as failed and all authorized formal jobs belong to execution_v2. No source, mask or training hyperparameter changed between these deployments.
