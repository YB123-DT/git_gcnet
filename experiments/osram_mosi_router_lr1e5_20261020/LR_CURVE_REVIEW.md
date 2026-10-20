# Evidence for the next bounded MOSI optimization candidate

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. Read-only inspection of the completed
original and low-LR training histories. No new training was authorized or started
by this review. Proposed3e-4 is a hypothesis, not a measured optimum.

Source histories on biggpu:

- /data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/{flat,joint}/history.json
- /data2/yb/remote_experiments/osram_mosi_router_lr1e5_20261020/execution_v1/runs/MOSI/seed_66/{flat,joint}/history.json

| Backbone LR | Model | Mean of last5 train MSE | Selected epochs by rate0.0–0.7 | Eight-rate BEST W-F1 |
|---|---|---:|---|---:|
| Original1e-3 first100,1e-4 last50 | Flat | 0.614963688 | 70,103,103,105,84,62,66,54 | 79.128027 |
| Constant1e-5 | Flat | 1.065611385 | 147,148,141,133,149,121,149,129 | 76.612759 |
| Original1e-3 first100,1e-4 last50 | Router | 0.573113027 | 149,149,118,142,123,83,46,47 | 79.012039 |
| Constant1e-5 | Router | 1.065819455 | 136,148,140,130,142,121,150,95 | 76.364318 |

Flat1e-5 epoch50/100/150 test eight-rate W-F1 was61.9629/72.5111/75.5859;
Router1e-5 was61.6699/72.5638/75.6860. These are same-epoch descriptive scores,
not the cumulative per-rate BEST above. Do not mix their selection meanings.

The late low-LR selected epochs, rising same-epoch scores and larger training
error support slow optimization within150epochs as an explanation. They do NOT
prove that longer training or a specific intermediate LR will improve W-F1,
nor establish that overfitting is absent. Task MSE is diagnostic, not the selection
criterion; original W-F1 and polarity thresholds remain unchanged.

One proposed next action, NOT LAUNCHED: seed66 Flat/Router matched pair,
150epochs from scratch, backbone constant3e-4; Router unchanged3e-4 first100,
3e-5 last50. Preserve original GCNet missing, exact common evaluation masks,
initialization, batch32, wd1e-5, dropout, MSE and per-rate Test-best W-F1.
No new module, feature or auxiliary loss. Compare both absolute score versus the
completed same-protocol original and Router-minus-Flat gain. An improvement to
both models is training optimization, not evidence for a Router contribution.
Do not automatically expand to more seeds without a positive screen.

Historical LR screens already exist: osram_lr_screen_mosi_20260907 used
H8/Output700 with JEPA/MMoE, while osram_lr_batch_screen_20261011 used the old
missing protocol and Flat/Nested (constant1e-4 scored79.516/79.711). They are
not matched evidence for this current GCNet-missing Flat/Router configuration;
do not rename them as current results or claim LR search is entirely new.
