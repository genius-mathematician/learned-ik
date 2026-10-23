"""Evaluate the trained IKNet against the exact analytical IK solver."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from train import IKNet, features_to_angles


def analytical_ik(X, L1, L2):
    """Closed-form 2-link IK, same elbow configuration as the dataset (t2 in [0, pi])."""
    x, y = X[:, 0], X[:, 1]
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    t2 = np.arccos(np.clip(c2, -1.0, 1.0))
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    return np.stack([t1, t2], axis=1)


def fk_np(theta, L1, L2):
    t1, t2 = theta[:, 0], theta[:, 1]
    return np.stack([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)], axis=1)


def wrap(a):
    """Wrap angle differences into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def stats(e):
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)),
            "max": float(e.max())}


def median_time(fn, repeats):
    fn()  # warm-up call, not timed
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X32, Y = d["X_test"], d["Y_test"].astype(np.float64)
    X = X32.astype(np.float64)
    Xt = torch.from_numpy(X32)

    ckpt = torch.load("models/iknet.pt")
    model = IKNet(hidden=ckpt["hidden"], scale=L1 + L2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---------- accuracy ----------
    with torch.inference_mode():
        theta_nn = features_to_angles(model(Xt)).numpy().astype(np.float64)
    theta_an = analytical_ik(X, L1, L2)

    pos_err_nn = np.linalg.norm(fk_np(theta_nn, L1, L2) - X, axis=1)
    pos_err_an = np.linalg.norm(fk_np(theta_an, L1, L2) - X, axis=1)
    ang_err_nn = np.degrees(np.abs(wrap(theta_nn - Y)))  # per joint, degrees

    summary = {
        "n_test": int(len(X)),
        "position_error_network": stats(pos_err_nn),
        "position_error_analytical": stats(pos_err_an),
        "angle_error_deg_network_joint1": stats(ang_err_nn[:, 0]),
        "angle_error_deg_network_joint2": stats(ang_err_nn[:, 1]),
    }

    print("Position error (link-length units), test set")
    print(f"{'':12s}{'mean':>10s}{'median':>10s}{'p95':>10s}{'p99':>10s}{'max':>10s}")
    for name in ["network", "analytical"]:
        s = summary[f"position_error_{name}"]
        print(f"{name:12s}" + "".join(f"{s[k]:10.2e}" for k in
                                      ["mean", "median", "p95", "p99", "max"]))

    # ---------- error vs distance from base ----------
    r = np.hypot(X[:, 0], X[:, 1])
    edges = np.linspace(abs(L1 - L2), L1 + L2, 9)
    bins = []
    print("\nNetwork error by reach r (singularities at both ends)")
    print(f"{'r range':>14s}{'count':>8s}{'mean':>10s}{'p99':>10s}{'max':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        e = pos_err_nn[m]
        bins.append({"r_lo": float(lo), "r_hi": float(hi), "count": int(m.sum()),
                     "mean": float(e.mean()), "p99": float(np.percentile(e, 99)),
                     "max": float(e.max())})
        print(f"{lo:6.2f}-{hi:5.2f}  {m.sum():8d}{e.mean():10.2e}"
              f"{np.percentile(e, 99):10.2e}{e.max():10.2e}")
    summary["network_error_by_radius"] = bins

    # ---------- speed ----------
    x1, xt1 = X[:1], Xt[:1]
    with torch.inference_mode():
        t_nn_batch = median_time(lambda: model(Xt), 50)
        t_nn_single = median_time(lambda: model(xt1), 2000)
    t_an_batch = median_time(lambda: analytical_ik(X, L1, L2), 50)
    t_an_single = median_time(lambda: analytical_ik(x1, L1, L2), 2000)

    summary["timing"] = {
        "torch_threads": torch.get_num_threads(),
        "batch_size": int(len(X)),
        "network_batch_ms": t_nn_batch * 1e3,
        "analytical_batch_ms": t_an_batch * 1e3,
        "network_single_us": t_nn_single * 1e6,
        "analytical_single_us": t_an_single * 1e6,
    }
    print(f"\nSpeed ({torch.get_num_threads()} CPU threads)")
    print(f"  batch of {len(X):,}: network {t_nn_batch*1e3:8.2f} ms | "
          f"analytical {t_an_batch*1e3:8.2f} ms")
    print(f"  single point:     network {t_nn_single*1e6:8.1f} us | "
          f"analytical {t_an_single*1e6:8.1f} us")

    # ---------- plots and saved results ----------
    out = Path("results")
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r, pos_err_nn, s=1, alpha=0.3, label="per test point")
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, [b["mean"] for b in bins], "r-o", label="bin mean")
    ax.set_yscale("log")
    ax.set_xlabel("reach r (distance from base)")
    ax.set_ylabel("position error")
    ax.set_title("Network IK error vs reach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_vs_reach.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(X[:, 0], X[:, 1], c=pos_err_nn, s=2, cmap="viridis",
                    norm=LogNorm(vmin=max(pos_err_nn.min(), 1e-5), vmax=pos_err_nn.max()))
    fig.colorbar(sc, ax=ax, label="position error")
    ax.set_aspect("equal")
    ax.set_title("Network IK error across the workspace")
    fig.tight_layout()
    fig.savefig(out / "error_map.png", dpi=120)

    with open(out / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSaved results/eval_summary.json, error_vs_reach.png, error_map.png")


if __name__ == "__main__":
    main()
