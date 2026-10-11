# Scheduling override: GPU7 drain, GPU1/2/3 three each

User asks GPU7 to finish current jobs without replacement, GPU1/2/3 to run
three jobs per card. New admissions only[1,2,3], max3; GPU4 prohibited, all
other GPUs excluded. GPU7 retains its current two children until completion.
Task scope stays47, IEMOCAP4Flat15 remain cancelled. No model/config changes.

Controller58b8c78, PID2220453, persistent tmux socketgcnet_complete150_20261011
session gpu123_three. Prior scheduler2177243 paused but not killed; terminal
sessions kept alive to avoid inherited-child hangups. All original8PIDs
preserved, no training restarted. New children have start_new_session=True.

GPU admission reserves9000MiB peak per child minus already allocated memory,
plus1500MiB extra slack; CPU-loading tasks still reserve their entire budget.
Cap3perGPU, unchanged disk/failure guards. Current membership/commands recorded
in QUEUE.json; GPU123_THREE.json stores adopted IDs/PIDs and source hash.
No future GPU7 admissions. Existing paused schedulers are not active queues.

Training outputs remain in the same isolated root; this is only a scheduling
change, not new experimental runs or additional seeds. Verify live progress
from QUEUE and histories before claiming completion or final performance.
