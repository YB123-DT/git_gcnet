# PAM-E smoke protocol

One real MOSI training batch was run on GPU with
`training_objective=pam-episodic-text`, `completion_path=pam-episodic-text`,
`train_rate_mode=fixed`, `fixed_missing_rate=0.7`, one optimizer step.

Observed:

- `classification_loss = 3.0074`, finite
- `jepa_loss = 0.5149`, finite
- `jepa_target_count = 491` real Text-missing targets
- one optimizer step completed
- all reported training metrics finite

Command path used `gcnet_missing_m3.train_gcnet.run_experiment` with a
one-batch wrapper so the real loader, model, OSRAM scan, CompletedReadFusion,
PAM ridge read, and backward pass all execute.  The full five-seed run was then
launched without waiting for further approval.
