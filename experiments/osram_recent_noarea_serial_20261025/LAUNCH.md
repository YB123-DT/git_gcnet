# Verified finite launch

Source814c5d9,immutable numerical sourcee0ca94c,cmerc-repro,biggpu.
Persistent tmux`recent_noarea_serial_20261025` launched at2026-10-25T16:47:45Z.
LAUNCH.json records command,GPUUUIDs,all deployed Python hashes and4real smokes.

Local38tests passed. Initial remote pytest collection lacked a reused test-only
fixture (`test_signature_order_audit_controls.py`);deployed that already-committed
fixture without changing any model/training/source-hashed computation. Remote
38tests then passed. The inherited optionaltorch-sparse import warning remains;
no package/environment was changed and the tested OSRAM path does not use it.

All4actual GPU smokes passed before formal admission:

| Configuration | Smoke seconds | Peak CUDA MiB | Peak RSS GiB |
| --- | ---: | ---: | ---: |
| Joint parallel |27.236|1285.979|2.540|
| Joint serial |25.219|1372.717|2.544|
| Frozen parallel |7.723|414.417|2.339|
| Frozen serial |7.625|525.449|2.335|

Initial predictions were exact originalFlat. Actual_scan probes verified one
Memoryscan per live forward. Jointserial correctly had zero new-block task
gradient on update1,nonzero on update2;frozenserial had nonzero task gradients
through the original frozenFlat. Selected checkpoints/RNG restored and exact
masked predictions replayed. Smoke does not establish performance improvement.

At16:50:32Z all18formaljobs were running,nofailed/pending jobs. Two150epoch
joint runs evaluate8rates;16frozenbranches cover2modes x8own-rateparents.
GPU0/1/2 only,6workers/card under real peak/resource admission. No otherjobs
signalled;no batch/feature/Missing/loss changes. All previous weights retained.

Remote root:
`/data2/yb/remote_experiments/osram_recent_noarea_serial_20261025/execution_v1`.
Controller log:sibling`controller_v1.log`. QUEUE.json/per-job SUMMARY.json are
live status authority. Final strict selected-prediction summary runs automatically
after all18complete. Stage1/stage2 are explicitly separate,not one mergedscore.
