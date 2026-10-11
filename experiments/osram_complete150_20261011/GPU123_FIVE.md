# Corrected concurrency: original2 + additional3 =5/GPU

User clarified additional3, not total3. Supersedes GPU123_THREE.md.
Only GPU1/2/3 admit new jobs, cap5each. GPU7 drains its original2jobs and
receives no replacements. Same47-task scope,15IEMOCAP4Flat remain cancelled.
No new experiments, configuration/seed/batch changes, or running-job restarts.

Observed allocations at change:1.8–5.3GiB per job, GPU1/2/3 free21.2/23.4/21.1GiB.
Reserve6000MiB per independent job plus1500MiB extra slack, subtract only the
unallocated reservation from actual free memory. CPU-loading jobs reserve full
6000MiB. Cap is an upper bound; wait if headroom insufficient. Five reservations
plus slack fit32GiB. Actual future fold peaks still require monitoring; do not
promise throughput improvement merely from additional concurrency.

Controller commitbb04539, PID2226416; tmux socketgcnet_complete150_20261011,
session gpu123_five. Old controller2220453 paused, not terminated; all adopted
jobs retain their PIDs. New children use independent sessions. GPU123_FIVE.json
records adopted PIDs and script hash. QUEUE.json is live scheduling authority.
Compile check and live cap5/whitelist verification passed; no dispatch traceback.
