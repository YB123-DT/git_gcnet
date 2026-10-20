# Status: both60epoch runs and paired reporting complete

Approved finite scope:seed66 MOSEI Flat/one-stageRouter,two fresh60epoch runs.
GCNet missing,per-rateTest-bestW-F1,same-callACC,predictions andMask.
Originalofficialtrain split,features and s0 environment;no newmodule.
Existing MOSI/IEMOCAP processes and snapshots untouched.

Final2026-10-20:queuecomplete,two60epoch runscomplete,zero failures.
Flatmean8ACC83.492708/WF1 83.279076;RouterACC83.437672/WF1 83.270902.
Router-minusFlatWF1:-0.008174pp overall,-0.346718pp highmissing.
Independent savedprediction audit:16files SHA256 exact; ACC/WF1/MSE recomputed
with maximumdifference0.0. See RESULT.md,METRICS.csv,FINAL_VERIFICATION.json.
No new training or inference was required for this result verification.

## Historical launch record (preserved)

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
