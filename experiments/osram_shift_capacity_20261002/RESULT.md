# Scalar filter capacity — all four complete

Seed66,100epochs, eight per-rate BEST checkpoints each. Internal Test-oracle diagnostics only.
GPU5 concurrent training; no model/data changes beyond declared scalar MLP depth/width.
No paired-view training or InfoNCE. Relation128, scalar output1 and zero-init residual unchanged.

|Filter|8-rate W-F1|High .5/.6/.7|Delta eight-rate vs Flat|Delta high vs Flat|
|---|---:|---:|---:|---:|
|Original Flat|81.068|76.352|—|—|
|Old D1-W128|80.497|75.813|-0.571|-0.540|
|D2-W128|80.284|75.508|-0.784|-0.845|
|D2-W256|80.164|74.843|-0.904|-1.509|
|D3-W128|80.270|75.582|-0.798|-0.771|
|D3-W256|80.932|76.200|-0.136|-0.152|

D3-W256 is best among these four and exceeds the old scalar residual by .435pp eight-rate
and .388pp high-missing, but remains below original Flat by .136pp and .152pp.
Depth/width effects are not monotonic. Single seed and four-config scan do not establish
stable gains or statistical significance. No automatic additional runs were launched.
