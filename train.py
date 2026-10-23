"""Train an MLP to approximate 2-link planar arm inverse kinematics."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 42
BATCH_SIZE = 1024
EPOCHS = 40
LR = 1e-3
HIDDEN = 256


# ---------- angle <-> network output conversion ----------
def angles_to_features(theta):
    """(N,2) angles -> (N,4) [cos t1, cos t2, sin t1, sin t2]."""
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)


def features_to_angles(f):
    """(N,4) network output -> (N,2) angles via atan2(sin, cos)."""
    return torch.atan2(f[:, 2:], f[:, :2])


def fk(theta, L1, L2):
    """Forward kinematics in torch, used to measure real position error."""
    t1, t2 = theta[:, 0], theta[:, 1]
    x = L1 * torch.cos(t1) + L2 * torch.cos(t1 + t2)
    y = L1 * torch.sin(t1) + L2 * torch.sin(t1 + t2)
    return torch.stack([x, y], dim=1)


# ---------- the model ----------
class IKNet(nn.Module):
    def __init__(self, hidden=HIDDEN, scale=1.8):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))  # saved with the model
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, xy):
        return self.net(xy / self.scale)  # normalise inputs to roughly [-1, 1]


@torch.no_grad()
def evaluate(model, X, Y, L1, L2):
    model.eval()
    pred = model(X)
    loss = nn.functional.mse_loss(pred, angles_to_features(Y)).item()
    pos_err = (fk(features_to_angles(pred), L1, L2) - X).norm(dim=1)
    return loss, pos_err.mean().item(), pos_err.max().item()


def train(hidden=HIDDEN, epochs=EPOCHS, model_path="models/iknet.pt",
          log_path="results/train_log.csv", verbose=True):
    """Train one IKNet and save the best checkpoint. Returns best val loss."""
    torch.manual_seed(SEED)

    d = np.load("data/ik_dataset.npz")
    L1, L2 = float(d["L1"]), float(d["L2"])
    X_train, Y_train = torch.from_numpy(d["X_train"]), torch.from_numpy(d["Y_train"])
    X_val, Y_val = torch.from_numpy(d["X_val"]), torch.from_numpy(d["Y_val"])
    F_train = angles_to_features(Y_train)  # targets precomputed once

    model = IKNet(hidden=hidden, scale=L1 + L2)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"IKNet: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    model_path, log_path = Path(model_path), Path(log_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val, best_mean = float("inf"), float("nan")
    g = torch.Generator().manual_seed(SEED)

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_mean_pos_err", "val_max_pos_err", "lr", "seconds"])

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            perm = torch.randperm(len(X_train), generator=g)
            total = 0.0
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                pred = model(X_train[idx])
                loss = loss_fn(pred, F_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / len(X_train)
            lr = opt.param_groups[0]["lr"]
            sched.step()
            secs = time.perf_counter() - t0

            val_loss, val_mean, val_max = evaluate(model, X_val, Y_val, L1, L2)
            writer.writerow([epoch, train_loss, val_loss, val_mean, val_max, lr, secs])
            fh.flush()

            flag = ""
            if val_loss < best_val:
                best_val, best_mean = val_loss, val_mean
                torch.save({"model_state": model.state_dict(), "hidden": hidden,
                            "L1": L1, "L2": L2}, model_path)
                flag = "  <- saved"
            if verbose:
                print(f"epoch {epoch:2d} | train {train_loss:.2e} | val {val_loss:.2e} | "
                      f"pos err mean {val_mean:.4f} max {val_max:.4f} | {secs:.1f}s{flag}")

    print(f"hidden={hidden}: {n_params:,} params | best val loss {best_val:.2e} | "
          f"val pos err mean {best_mean:.4f} | saved {model_path}")
    return best_val


def main():
    p = argparse.ArgumentParser(description="Train IKNet on the 2-link IK dataset.")
    p.add_argument("--hidden", type=int, default=HIDDEN, help="neurons per hidden layer")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--model-path", default="models/iknet.pt")
    p.add_argument("--log-path", default="results/train_log.csv")
    a = p.parse_args()
    train(a.hidden, a.epochs, a.model_path, a.log_path)


if __name__ == "__main__":
    main()
