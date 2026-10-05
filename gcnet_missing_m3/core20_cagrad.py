"""CAGrad shared-gradient direction, without replacing the model's optimizer.

Paper: https://arxiv.org/abs/2110.14048v2, Eq. (3), Algorithm 1.
Author reference: Cranial-XIX/CAGrad, dc3d48152b6196945cfd56144879b9d42353b095,
nyuv2/utils.py::cagrad. The simplex solve and rescale modes follow that core;
we solve the unsmoothed paper objective with explicit convergence checks.
Only shared parameters belong in ``grads``. Task-specific parameters retain
their ordinary gradients of the mean task loss (including unused-task zeros).
The caller assigns the direction to .grad, then clips and steps existing Adam.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize


def conflict_averse_direction(
    grads: torch.Tensor,
    alpha: float = 0.4,
    rescale: int = 1,
    *,
    maxiter: int = 200,
    ftol: float = 1e-12,
    convergence_tol: float = 1e-6,
):
    """Return ``(direction[P], diagnostics)`` for task gradients ``[P, T]``.

    ``alpha`` is the paper's conflict radius c, not a learning rate. With
    A=G.T G, b=1/T and r=alpha*||G b||, solve
    min_{w>=0, sum(w)=1} w.T A b + r sqrt(w.T A w), then construct
    d=G b + r G w / ||G w||. Author rescale modes are 0: d,
    1: d/(1+alpha**2), 2: d/(1+alpha). The paper's ball and duality checks
    concern d before this positive rescaling.

    A common gradient scale is removed for the double precision CPU solve;
    this preserves the minimizer and avoids scale-dependent stopping tests.
    Failed/nonoptimal solves raise RuntimeError, never fall back to mean.
    Returned diagnostics are JSON-compatible and contain normalized residuals.
    No graph is retained and input gradients are never changed.
    """
    if not isinstance(grads, torch.Tensor) or grads.ndim != 2:
        raise ValueError("grads must be a floating Tensor[P, T]")
    if not grads.is_floating_point() or min(grads.shape) == 0:
        raise ValueError("grads must be floating with nonempty P and T")
    if not bool(torch.isfinite(grads).all()):
        raise ValueError("grads must be finite")
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    if not isinstance(rescale, int) or rescale not in (0, 1, 2):
        raise ValueError("rescale must be 0, 1, or 2")
    if maxiter < 1 or not 0 < ftol < 1 or not 0 < convergence_tol < 1:
        raise ValueError("invalid solver stopping settings")

    g = grads.detach().to(device="cpu", dtype=torch.float64).numpy()
    tasks = g.shape[1]
    uniform = np.full(tasks, 1.0 / tasks)
    # Entry-wise scaling avoids overflow while computing the column norms.
    entry_scale = float(np.max(np.abs(g)))
    scale = entry_scale if entry_scale else 1.0
    g = g / scale
    mean = g @ uniform
    mean_norm = float(np.linalg.norm(mean))
    radius = alpha * mean_norm
    gram = g.T @ g
    linear = gram @ uniform

    def objective(w):
        return float(w @ linear + radius * np.linalg.norm(g @ w))

    def derivative(w):
        gw = g @ w
        norm = float(np.linalg.norm(gw))
        # At gw=0 this chooses the valid zero subgradient of the norm.
        return linear + (radius * (g.T @ gw) / norm if norm else 0.0)

    if alpha == 0 or mean_norm == 0:
        weights = uniform
        direction = mean
        status = "alpha_zero" if alpha == 0 else "mean_zero"
        iterations = 0
        solver_message = "exact closed form; no dual solve required"
        residual = gap = 0.0
    else:
        result = minimize(
            objective,
            uniform,
            jac=derivative,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * tasks,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                          "jac": lambda w: np.ones_like(w)}],
            options={"maxiter": maxiter, "ftol": ftol},
        )
        weights = result.x
        if not result.success or not np.isfinite(weights).all():
            raise RuntimeError(f"CAGrad simplex solve failed: {result.message}")
        feasibility = max(abs(float(weights.sum()) - 1.0),
                          max(0.0, -float(weights.min())),
                          max(0.0, float(weights.max()) - 1.0))
        gw = g @ weights
        gw_norm = float(np.linalg.norm(gw))
        direction = mean + (radius * gw / gw_norm if gw_norm else 0.0)
        # Direction supplies a valid dual subgradient, including gw=0.
        dual_gradient = g.T @ direction
        multiplier = float(weights @ dual_gradient)
        reduced = dual_gradient - multiplier
        # Simplex KKT: reduced costs >=0; w_i * reduced_i =0.
        residual = max(max(0.0, -float(reduced.min())),
                       float(np.max(np.abs(weights * reduced))))
        gap = max(0.0, objective(weights) - float(dual_gradient.min()))
        denominator = max(1.0, float(np.max(np.abs(gram))))
        residual /= denominator
        gap /= denominator
        if feasibility > convergence_tol or residual > convergence_tol or gap > convergence_tol:
            raise RuntimeError(
                f"CAGrad solve not optimal: feasibility={feasibility:.3g}, "
                f"KKT={residual:.3g}, duality_gap={gap:.3g}"
            )
        status = "converged"
        iterations = int(result.nit)
        solver_message = str(result.message)

    feasibility = max(abs(float(weights.sum()) - 1.0),
                      max(0.0, -float(weights.min())),
                      max(0.0, float(weights.max()) - 1.0))
    divisor = (1.0, 1.0 + alpha ** 2, 1.0 + alpha)[rescale]
    output = torch.from_numpy(np.asarray(direction * scale / divisor)).to(grads)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("CAGrad direction overflowed the input dtype")
    diagnostics = {
        "status": status,
        "solver": "scipy.optimize.minimize/SLSQP",
        "solver_message": solver_message,
        "iterations": iterations,
        "weights": weights.tolist(),
        "alpha": float(alpha),
        "rescale": int(rescale),
        "gradient_scale": scale,
        "normalized_dual_objective": objective(weights),
        "normalized_uniform_objective": objective(uniform),
        "simplex_residual": feasibility,
        "kkt_residual": residual,
        "normalized_duality_gap": gap,
        "normalized_ball_violation": max(0.0, float(np.linalg.norm(direction - mean)) - radius),
        "convergence_tol": convergence_tol,
    }
    return output, diagnostics
