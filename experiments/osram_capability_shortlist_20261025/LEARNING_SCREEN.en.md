# Learning-mechanism screening: check existing counterevidence first

No new training or diagnostic was run. The existing text-condition audit (20261022/RESULT.md) examined 64 paired targets per dataset with fixed history and dropout disabled. MOSI/MOSEI negative task-gradient counts (16/11) coincided with opposite scalar MSE residual signs. Overall prediction-Jacobian cosine means were 0.861/0.868, with no negative pairs. A Text-present SGD update improved the Text-missing target loss in 47/53 pairs. This does not support presupposing pervasive representation conflict.

Limitations: s0 environment, one seed66 miss0.3 checkpoint per dataset, single-target finite updates, not current cmerc training trajectories or cross-example Adam behavior. This neither excludes every optimization conflict nor justifies repeating the same diagnostic.

core20.py constructs C20 CAGrad tasks from local/base/full exits, not availability groups. Changing task grouping is a distinct comparison, not a new optimizer.

MLA (CVPR2024), MMPareto (ICML2024), and CAGrad (NeurIPS2021) differ in objectives and affected parameters. MMPareto targets unimodal auxiliary versus multimodal objective conflicts. Deprioritize conflict protection until project evidence warrants it. Further search should distinguish knowledge transfer, learning schedules, and conflict handling, with nearest-history and budget controls. Better-observed views are not automatically more accurate per example.

Sources:
- https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html
- https://proceedings.mlr.press/v235/wei24d.html
- https://proceedings.nips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

