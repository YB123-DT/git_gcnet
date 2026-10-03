# Locked twenty-candidate screen

INTERNAL DIAGNOSTIC ONLY

WITHDRAWN BY USER BEFORE TRAINING. Retained as source-verification history, not an
active recommendation or authorized launch list.

All sources are vision, VQA or recommendation, not MSA/MERC. These are explicit
OSRAM adaptations, not full-paper reproductions or promised improvements. Run one
fixed seed66/100epoch configuration per method with the same masked128-dimensional
interface and zero-start output residual; preserve Flat, Memory and task loss.

|#|Candidate and primary source|Tested mechanism|Added parameters|
|---:|---|---|---:|
|1|[film](https://doi.org/10.1609/aaai.v32i1.11671)|Local-conditioned affine memory modulation|308680|
|2|[se](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)|Masked slot squeeze/excitation|307528|
|3|[eca](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)|Adjacent feature-channel convolution|305349|
|4|[cbam](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)|Channel gate then fixed-slot gate|307542|
|5|[gct](https://arxiv.org/abs/1909.11519)|Normalized L2 channel transformation|305728|
|6|[simam](https://proceedings.mlr.press/v139/yang21o.html)|Masked slot-energy weights|305344|
|7|[sk](https://arxiv.org/abs/1903.06586)|Two dense branches with channel-wise selection|350944|
|8|[mlb](https://arxiv.org/abs/1610.04325)|Low-rank Hadamard bilinear|354880|
|9|[mfb](https://arxiv.org/abs/1708.01471)|Four-factor product pooling and normalization|453952|
|10|[mutan](https://arxiv.org/abs/1705.06676)|Two-stage Tucker factors, rank4|486976|
|11|[block](https://arxiv.org/abs/1902.00038)|Four block chunks, rank4 each|388672|
|12|[mcb](https://doi.org/10.18653/v1/D16-1044)|Fixed1024 CountSketch and FFT|436544|
|13|[ban](https://arxiv.org/abs/1805.07932)|Single-query bilinear evidence attention|388032|
|14|[dcnv2](https://arxiv.org/abs/2008.13535)|Two rank32 cross layers|470592|
|15|[cin](https://arxiv.org/abs/1803.05170)|Two16-map vector-wise cross layers|311248|
|16|[autoint](https://arxiv.org/abs/1810.11921)|Two two-head token interaction layers|437440|
|17|[din](https://arxiv.org/abs/1706.06978)|Local-conditioned unnormalized activation pooling|382561|
|18|[dlrm](https://arxiv.org/abs/1906.00091)|Ten pair dots concatenated with Local|323136|
|19|[aff](https://arxiv.org/abs/2009.14082)|Pooled Local/history channel blend|322048|
|20|[nonlocal](https://arxiv.org/abs/1711.07971)|Signed dot/active-count token interaction|338432|

Flat has13,509,793 parameters. Added counts include shared input/type/output
adaptation, not just the cited operator. No Flat freezing or exact parameter
matching is claimed; SimAM's operator is parameter-free, its wrapper is not.
Only forward512 history is used. Sanitize inactive/padding tokens before and after
biased operations; skip the first valid utterance residual. Zero-start type
embeddings, no new stochastic dropout. Non-BAN bilinear operators use active-history
mean; BAN retains evidence slots.

See paper_bank.json and literature/*.json for method/code/attribution evidence.
Original SK code access is404; confirmed original fork and coauthor adaptation were
checked. DCNv2 code is official Google code, without personal-author attribution.
Some licenses are absent/inconsistent: independently implement equations, do not
copy code. DIN/AFF paper-code differences are explicit; AFF uses pooled readouts,
not full spatial MS-CAM. SK's dense branches and CBAM's typed-slot convolution are
explicit domain adaptations. Related mechanisms are not claimed as twenty novel
ideas, nor are width variants counted separately. Do not change candidates based
on test outcomes. Per-rate Test-oracle plus best-of20 selection is biased internal
screening, not a formal paper result.
