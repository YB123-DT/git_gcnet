# C18: author-order Group DRO transfer

Primary source: [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731).
Author implementation: [LossComputer.compute_robust_loss](https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/loss.py).

Seven availability groups A/T/V/AT/AV/TV/ATV. Each step computes mean original
task loss for present groups; absent groups have zero group loss. Update ALL
seven probabilities using q <- softmax(log(q) + eta * group_loss.detach()),
then compute the current robust task loss sum_g q_g * group_loss_g. Log-space
arithmetic changes numerical stability, not the exponentiated-gradient rule.
No present-group renormalization; no arbitrary loss clipping.

The earlier pattern-groupdro implementation remains unchanged under its old
flag. It used previous weights and present-group renormalization, so this new
run is explicitly a source-order/protocol comparison, not a never-tried idea.

Retain cfg84 eta=.1 and Adam weight decay=1e-5 as a fixed transfer configuration.
No claim these are the paper's best regularization recipe. Smaller weighted
loss when probability mass belongs to absent groups is intentional author
behavior, and changes effective gradient magnitude versus sample-mean.

Full continuation saves/restores the seven probabilities in auxiliary_state.
Older Group DRO last_training files without this state fail recovery clearly;
they are not silently called complete continuations.

Status: implemented, author-order and recovery tests passed; no score yet.
