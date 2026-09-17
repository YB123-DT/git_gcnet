# PAM-T smoke protocol

Run one real MOSI train batch with `training_objective=pam-text` and verify:

- forward/backward is finite;
- Text-missing prediction count is nonzero;
- PAM source encoder and beta receive gradients;
- Teacher updates only after the optimizer step;
- predicted Text is used only through `read_node`, while `write_node` remains the observed node.

The complete five-seed run is launched only after this real-batch check.
