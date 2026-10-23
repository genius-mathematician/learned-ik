"""Hybrid IK: network initial guess + damped least-squares (Jacobian) refinement."""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import IKNet, features_to_angles
from evaluate import analytical_ik, fk_np, wrap, median_time

TOL = 1e-10      # converged when the hand is within this distance of the target
MAX_ITERS = 100
DAMPING = 1e-3   # lambda in damped least squares; keeps steps finite near singularities


def jacobian(theta, L1, L2):
    """Batch of 2x2 Jacobians d(x, y)/d(t1, t2), shape (N, 2, 2)."""
    t1, t2 = theta[:, 0], theta[:, 1]
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    J = np.empty((len(theta), 2, 2))
    J[:, 0, 0] = -L1 * s1 - L2 * s12
    J[:, 0, 1] = -L2 * s12
    J[:, 1, 0] = L1 * c1 + L2 * c12
    J[:, 1, 1] = L2 * c12
    return J


def refine(theta0, X, L1, L2, tol=TOL, max_iters=MAX_ITERS, lam=DAMPING):
    """Damped least-squares IK on every point at once.
    Returns final angles and the iteration count each point needed."""
    theta = theta0.copy()
    iters = np.full(len(X), max_iters)
    active = np.ones(len(X), dtype=bool)
    eye = np.eye(2)
    for k in range(max_iters + 1):
        err = X - fk_np(theta, L1, L2)
        done = active & (np.linalg.norm(err, axis=1) < tol)
        iters[done] = k
        active &= ~done
        if not active.any() or k == max_iters:
            break
        J = jacobian(theta[active], L1, L2)
        A = J @ J.transpose(0, 2, 1) + lam**2 * eye              # J J^T + lambda^2 I
        y = np.linalg.solve(A, err[active][..., None])[..., 0]    # solve A y = e
        theta[active] += np.einsum("nji,nj->ni", J, y)            # step = J^T y
    return theta, iters


def run_stats(theta, iters, X, L1, L2):
    err = np.linalg.norm(fk_np(theta, L1, L2) - X, axis=1)
    conv = err < TOL
    it = iters[conv]
    return {
        "converged_pct": float(100 * conv.mean()),
        "iters_mean": float(it.mean()),
        "iters_median": float(np.median(it)),
        "iters_p99": float(np.percentile(it, 99)),
        "iters_max": int(it.max()),
        "wrong_elbow": int((wrap(theta[:, 1]) < 0).sum()),
        "final_err_max": float(err.max()),
    }


def main():
    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X = d["X_test"].astype(np.float64)
    Xt = torch.from_numpy(d["X_test"])

    ckpt = torch.load("models/iknet.pt")
    model = IKNet(hidden=ckpt["hidden"], scale=L1 + L2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def network_guess():
        with torch.inference_mode():
            return features_to_angles(model(Xt)).numpy().astype(np.float64)

    cold_start = np.tile([0.0, np.pi / 2], (len(X), 1))

    theta_cold, it_cold = refine(cold_start, X, L1, L2)
    theta_hyb, it_hyb = refine(network_guess(), X, L1, L2)

    results = {
        "cold_start": run_stats(theta_cold, it_cold, X, L1, L2),
        "hybrid": run_stats(theta_hyb, it_hyb, X, L1, L2),
    }
    results["cold_start"]["batch_ms"] = 1e3 * median_time(
        lambda: refine(cold_start, X, L1, L2), 5)
    results["hybrid"]["batch_ms"] = 1e3 * median_time(
        lambda: refine(network_guess(), X, L1, L2), 5)
    results["analytical_batch_ms"] = 1e3 * median_time(
        lambda: analytical_ik(X, L1, L2), 20)

    print(f"Jacobian IK on {len(X):,} test points (tol {TOL:g}, max {MAX_ITERS} iters)")
    print(f"{'method':12s}{'conv %':>8s}{'it mean':>9s}{'median':>8s}{'p99':>6s}"
          f"{'max':>6s}{'wrong elbow':>13s}{'time ms':>10s}")
    for name in ["cold_start", "hybrid"]:
        r = results[name]
        print(f"{name:12s}{r['converged_pct']:8.2f}{r['iters_mean']:9.2f}"
              f"{r['iters_median']:8.0f}{r['iters_p99']:6.0f}{r['iters_max']:6d}"
              f"{r['wrong_elbow']:13d}{r['batch_ms']:10.2f}")
    print(f"analytical (reference): {results['analytical_batch_ms']:.2f} ms")

    out = Path("results")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, MAX_ITERS + 2) - 0.5
    ax.hist(it_cold, bins=bins, alpha=0.6, label="cold start")
    ax.hist(it_hyb, bins=bins, alpha=0.6, label="hybrid (network guess)")
    ax.set_yscale("log")
    ax.set_xlabel("iterations to converge")
    ax.set_ylabel("number of test points")
    ax.set_title("Jacobian IK iterations: cold start vs network warm start")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "iterations_hist.png", dpi=120)

    with open(out / "hybrid_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("Saved results/hybrid_summary.json, iterations_hist.png")


if __name__ == "__main__":
    main()
