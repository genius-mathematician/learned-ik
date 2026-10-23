"""Line-by-line profile of the Jacobian refinement loop."""
import numpy as np
import torch
from line_profiler import LineProfiler

from train import features_to_angles
from sweep import load_model
from hybrid import refine

N_RUNS = 20


def main():
    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X = d["X_test"].astype(np.float64)
    Xt = torch.from_numpy(d["X_test"])

    model = load_model("models/iknet_h32.pt", L1, L2)
    with torch.inference_mode():
        theta0 = features_to_angles(model(Xt)).numpy().astype(np.float64)

    refine(theta0, X, L1, L2)  # warm-up, not profiled

    lp = LineProfiler(refine)
    for _ in range(N_RUNS):
        lp.runcall(refine, theta0, X, L1, L2)
    lp.print_stats(output_unit=1e-3)  # report times in milliseconds


if __name__ == "__main__":
    main()
