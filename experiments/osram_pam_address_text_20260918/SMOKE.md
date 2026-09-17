# PAM-A smoke protocol

One real MOSI training batch on GPU with:

- `training_objective=pam-address-text`
- `completion_path=pam-address-text`
- `train_rate_mode=fixed`, `fixed_missing_rate=0.7`
- one optimizer step
- `lambda_a=5.0`

Observed after the step:

- `classification_loss = 3.0793`
- `address_loss = 0.0264`
- `jepa_target_count = 2925` real address targets
- `loss = 3.2113`
- all metrics finite and the optimizer step completed

The full five-seed run was launched immediately after this check.
