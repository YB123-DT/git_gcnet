# OSRAM learning-rate screen on CMU-MOSI

This is a narrow optimization diagnostic for the already implemented OSRAM
H8/Output700 backbone. No model module, loss, mask protocol, feature, or
evaluation rule is changed.

The current reference is OSRAM H8/Output700 with learning rate 1e-3. The
screen tests only two lower learning rates:

| Variant | Learning rate | Fixed backbone |
|---|---:|---|
| LR5e-4 | 5e-4 | 8 heads, key/value 32, output 700 |
| LR3e-4 | 3e-4 | 8 heads, key/value 32, output 700 |

The lower rates are tested because the existing mixed-rate MOSI training
objective combines emotion regression with JEPA gradients, and the previous
learning-rate screen indicated that 1e-3 can be aggressive for this setting.
This is a falsifiable optimization check, not a new method claim.
