# Status: realGPU smoke passed; two formal processes launched

Approved finite scope:seed66 MOSEI Flat/one-stageRouter,two fresh60epoch runs.
GCNet missing,per-rateTest-bestW-F1,same-callACC,predictions andMask.
Originalofficialtrain split,features and s0 environment;no newmodule.
Existing MOSI/IEMOCAP processes and snapshots untouched.

53related tests passed; compileall and diff checks passed. RealGPU3batchJoint
smoke completed with finite/nonzero backbone/Router gradients, parameter update,
full model/Adam/RNG recovery and exact selectedprediction replay.
Peak allocated1527.58056640625MiB. All68,569inputfeature/label hashes matched.
Smoke is correctness verification,not a performance result.

Immutable sourcecommit51a29dff5a2c22363900f9071bc48a024b039fcc.
Archive SHA25650404082d11ec4f6970d7124f8c395d0bfe3886f791e46155bffc9377d33de50.
Serverbiggpu,physicalGPU1 UUID`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
Source:`/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/source_51a29df`.
Output:`/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1`.
tmux:`mosei_router_51a29df`;controllerPID846986.
Two formalworkerPIDs:Flat850573,Router850657. At the recorded launch inspection
they were still initializing,not completedepochs. See VERIFICATION_AT_LAUNCH.json.
Subsequent remoteinspection confirmed both workerstatus=running and effective
CMUMOSEI/60epochs/lr.001/GCNet-v1/per-rateTest-bestweighted_f1 configs;
no errors recorded. An epoch completion or finalscore is not yet claimed.
The queue will train60epochs and automatically produce paired finalmetrics.
No performance improvement claimed; no extra seeds or methods queued.
