"""Network size sweep: network accuracy vs refinement iterations vs total hybrid time."""
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import IKNet, features_to_angles, train
from evaluate import fk_np, median_time
from hybrid import refine, run_stats

SIZES = [16, 32, 64, 128, 256]
EPOCHS = 40


def load_model(path, L1, L2):
    ckpt = torch.load(path)
    model = IKNet(hidden=ckpt["hidden"], scale=L1 + L2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X = d["X_test"].astype(np.float64)
    Xt = torch.from_numpy(d["X_test"])

    rows = []
    for h in SIZES:
        path = Path(f"models/iknet_h{h}.pt")
        if not path.exists():  # skip training if this size was already trained
            train(hidden=h, epochs=EPOCHS, model_path=path,
                  log_path=f"results/train_log_h{h}.csv", verbose=False)
        model = load_model(path, L1, L2)

        def forward():
            with torch.inference_mode():
                return model(Xt)

        theta0 = features_to_angles(forward()).numpy().astype(np.float64)
        net_err = np.linalg.norm(fk_np(theta0, L1, L2) - X, axis=1)
        theta, iters = refine(theta0, X, L1, L2)
        s = run_stats(theta, iters, X, L1, L2)

        net_ms = 1e3 * median_time(forward, 30)
        refine_ms = 1e3 * median_time(lambda: refine(theta0, X, L1, L2), 5)

        rows.append({
            "hidden": h,
            "params": sum(p.numel() for p in model.parameters()),
            "net_err_mean": float(net_err.mean()),
            "net_err_max": float(net_err.max()),
            "converged_pct": s["converged_pct"],
            "iters_mean": s["iters_mean"],
            "iters_max": s["iters_max"],
            "wrong_elbow": s["wrong_elbow"],
            "net_ms": net_ms,
            "refine_ms": refine_ms,
            "total_ms": net_ms + refine_ms,
        })

    print(f"\n{'hidden':>6s}{'params':>9s}{'net err':>10s}{'conv %':>8s}{'it mean':>8s}"
          f"{'it max':>7s}{'wrong':>6s}{'net ms':>8s}{'refine':>8s}{'total':>8s}")
    for r in rows:
        print(f"{r['hidden']:6d}{r['params']:9,d}{r['net_err_mean']:10.2e}"
              f"{r['converged_pct']:8.2f}{r['iters_mean']:8.2f}{r['iters_max']:7d}"
              f"{r['wrong_elbow']:6d}{r['net_ms']:8.2f}{r['refine_ms']:8.2f}"
              f"{r['total_ms']:8.2f}")

    out = Path("results")
    with open(out / "size_sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(r["hidden"]) for r in rows]
    net = [r["net_ms"] for r in rows]
    ref = [r["refine_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, net, label="network forward pass")
    ax.bar(labels, ref, bottom=net, label="Jacobian refinement")
    for i, r in enumerate(rows):
        ax.text(i, r["total_ms"], f"{r['iters_mean']:.1f} it", ha="center", va="bottom")
    ax.set_xlabel("hidden layer width")
    ax.set_ylabel("time for 20,000 points (ms)")
    ax.set_title("Hybrid IK time vs network size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "size_sweep.png", dpi=120)
    print("\nSaved results/size_sweep.csv, size_sweep.png")


if __name__ == "__main__":
    main()
