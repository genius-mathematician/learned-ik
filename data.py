"""Generate a dataset for learning 2-link planar arm inverse kinematics."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt

L1, L2 = 1.0, 0.8  # link lengths
SEED = 42          # fixed seed so the dataset is identical every run


def forward_kinematics(theta1, theta2):
    """Joint angles -> end-effector position (x, y)."""
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y


def make_split(n, rng):
    """Sample n random arm poses and return (positions, angles)."""
    theta1 = rng.uniform(-np.pi, np.pi, n)
    theta2 = rng.uniform(0.01, np.pi - 0.01, n)  # one elbow configuration only
    x, y = forward_kinematics(theta1, theta2)
    inputs = np.stack([x, y], axis=1).astype(np.float32)
    targets = np.stack([theta1, theta2], axis=1).astype(np.float32)
    return inputs, targets


def main():
    rng = np.random.default_rng(SEED)
    sizes = {"train": 200_000, "val": 20_000, "test": 20_000}

    arrays = {}
    for name, n in sizes.items():
        X, Y = make_split(n, rng)
        arrays[f"X_{name}"] = X
        arrays[f"Y_{name}"] = Y
        r = np.hypot(X[:, 0], X[:, 1])
        print(f"{name:5s}: {n:7,d} samples | reach r in [{r.min():.3f}, {r.max():.3f}]")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "ik_dataset.npz", L1=L1, L2=L2, **arrays)
    print(f"Saved {out_dir / 'ik_dataset.npz'}")

    # Sanity-check plot: training targets should fill an annulus
    pts = arrays["X_train"][:5000]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title("Training targets (first 5,000)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(out_dir / "workspace.png", dpi=120)
    print(f"Saved {out_dir / 'workspace.png'}")


if __name__ == "__main__":
    main()
