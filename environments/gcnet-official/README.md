# GCNet official-declared environment

This environment follows the top-level versions declared in the upstream
GCNet README:

- Python 3.8
- CUDA 10.2
- PyTorch 1.8.0
- torchvision 0.9.0
- PyTorch Geometric 2.0.1

The upstream repository does not provide a complete lock file. Compatible PyG
extension versions are therefore pinned here to the wheels published for
PyTorch 1.8.0 + CUDA 10.2.

The verified remote interpreter is:

```text
/data2/yb/reproduction_envs/gcnet-official/bin/python
```

Install PyG extensions from the matching wheel index:

```bash
pip install --no-deps \
  torch-scatter==2.0.8 \
  torch-sparse==0.6.12 \
  torch-cluster==1.5.9 \
  torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-geometric==2.0.1
```

Verified on a Tesla V100-SXM2-32GB with the repository's RGCNConv and
GraphConv CUDA forward pass and the full unit-test suite.

Exact verified versions on `biggpu` are Python 3.8.20, PyTorch 1.8.0,
CUDA 10.2, cuDNN 7605, PyG 2.0.1, NumPy 1.21.6, SciPy 1.7.3,
scikit-learn 1.0.2, and NVIDIA driver 575.51.03.

Strict deterministic mode is not a usable training mode for this model:
`torch_scatter.scatter_add_cuda_kernel`, reached through PyG graph aggregation,
has no deterministic CUDA implementation in this stack. The unified protocol
therefore isolates every controllable random stream and uses paired repeated
runs; it does not claim bitwise-identical GPU trajectories.
