"""Optimized damped least-squares refinement, verified against the original."""
import json
from pathlib import Path

import numpy as np
import torch

from train import features_to_angles
from sweep import load_model
from evaluate import fk_np, wrap, median_time
from hybrid import refine, run_stats, TOL, MAX_ITERS, DAMPING


def refine_fast(theta0, X, L1, L2, tol=TOL, max_iters=MAX_ITERS, lam=DAMPING):
    """Same algorithm as hybrid.refine, rewritten for speed:
    - converged points are removed from the working arrays
    - sin/cos computed once per pass, shared by FK and the Jacobian
    - 2x2 damped solve written out in closed form (no matrix routines)"""
    theta = theta0.copy()
    iters = np.full(len(X), max_iters)
    idx = np.arange(len(X))                    # original index of each active point
    t1, t2 = theta[:, 0].copy(), theta[:, 1].copy()
    tx, ty = X[:, 0].copy(), X[:, 1].copy()
    tol2, lam2 = tol * tol, lam * lam

    for k in range(max_iters + 1):
        s1, c1 = np.sin(t1), np.cos(t1)
        s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
        ex = tx - (L1 * c1 + L2 * c12)         # position error, x
        ey = ty - (L1 * s1 + L2 * s12)         # position error, y

        done = ex * ex + ey * ey < tol2        # squared distance: no sqrt needed
        if done.any():
            iters[idx[done]] = k
            theta[idx[done], 0] = t1[done]
            theta[idx[done], 1] = t2[done]
            keep = ~done
            idx, t1, t2, tx, ty = idx[keep], t1[keep], t2[keep], tx[keep], ty[keep]
            s1, c1, s12, c12 = s1[keep], c1[keep], s12[keep], c12[keep]
            ex, ey = ex[keep], ey[keep]
        if len(idx) == 0 or k == max_iters:
            break

        # Jacobian entries, reusing the trig values above
        J00 = -L1 * s1 - L2 * s12
        J01 = -L2 * s12
        J10 = L1 * c1 + L2 * c12
        J11 = L2 * c12

        # A = J J^T + lambda^2 I  is symmetric:  [[a, b], [b, c]]
        a = J00 * J00 + J01 * J01 + lam2
        b = J00 * J10 + J01 * J11
        c = J10 * J10 + J11 * J11 + lam2
        det = a * c - b * b

        # y = A^-1 e  (closed-form 2x2 inverse)
        y0 = (c * ex - b * ey) / det
        y1 = (a * ey - b * ex) / det

        # step = J^T y
        t1 = t1 + J00 * y0 + J10 * y1
        t2 = t2 + J01 * y0 + J11 * y1

    theta[idx, 0], theta[idx, 1] = t1, t2      # any points that never converged
    return theta, iters


def main():
    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X = d["X_test"].astype(np.float64)
    Xt = torch.from_numpy(d["X_test"])

    model = load_model("models/iknet_h32.pt", L1, L2)
    with torch.inference_mode():
        theta0 = features_to_angles(model(Xt)).numpy().astype(np.float64)

    # ---------- 1. verify: same answers as the original ----------
    theta_old, it_old = refine(theta0, X, L1, L2)
    theta_new, it_new = refine_fast(theta0, X, L1, L2)

    same_iters_pct = 100 * float((it_old == it_new).mean())
    max_angle_diff = float(np.abs(wrap(theta_old - theta_new)).max())
    s_old = run_stats(theta_old, it_old, X, L1, L2)
    s_new = run_stats(theta_new, it_new, X, L1, L2)

    print("Verification (20,000 test points, width-32 network guesses)")
    print(f"  identical iteration counts: {same_iters_pct:.2f}%")
    print(f"  max angle difference:       {max_angle_diff:.2e} rad")
    for name, s in [("original", s_old), ("fast", s_new)]:
        print(f"  {name:9s} conv {s['converged_pct']:.2f}% | it mean {s['iters_mean']:.2f}"
              f" | wrong elbow {s['wrong_elbow']} | max err {s['final_err_max']:.1e}")

    # ---------- 2. benchmark ----------
    t_old = 1e3 * median_time(lambda: refine(theta0, X, L1, L2), 20)
    t_new = 1e3 * median_time(lambda: refine_fast(theta0, X, L1, L2), 20)
    print("\nRefinement time, median of 20 runs")
    print(f"  original: {t_old:7.2f} ms")
    print(f"  fast:     {t_new:7.2f} ms   ({t_old / t_new:.1f}x faster)")

    out = {"same_iters_pct": same_iters_pct, "max_angle_diff_rad": max_angle_diff,
           "original": {**s_old, "refine_ms": t_old},
           "fast": {**s_new, "refine_ms": t_new},
           "speedup": t_old / t_new}
    with open(Path("results") / "refine_speedup.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved results/refine_speedup.json")


if __name__ == "__main__":
    main()
