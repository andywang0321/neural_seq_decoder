#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Professional NeurIPS-like style
# -------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})


# Main academic colors
COL_PRIMARY = "firebrick"
COL_SECONDARY = "slategray"
COL_TERTIARY = "steelblue"


def moving_average(x, window):
    if window <= 1 or len(x) == 0:
        return x
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad for equal-length vector
    return np.concatenate([np.full(window - 1, np.nan), ma])


def load_stats(run_dir):
    stats_path = os.path.join(run_dir, "trainingStats")
    if not os.path.exists(stats_path):
        print(f"[WARN] No trainingStats found in {run_dir}, skipping.")
        return None, None
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return np.array(stats.get("testLoss", [])), np.array(stats.get("testCER", []))


# -------------------------------
# Plotting functions
# -------------------------------

def plot_loss(run_dir, steps, loss_raw, loss_smooth=None):
    plt.figure(figsize=(6, 4))                                # <-- larger figure
    eps = 1e-8
    loss = np.maximum(loss_raw, eps)

    plt.plot(steps, loss, color=COL_PRIMARY, linewidth=1.8, label="Test CTC loss")
    if loss_smooth is not None:
        plt.plot(steps, np.maximum(loss_smooth, eps), color=COL_SECONDARY,
                 linewidth=2.2, label="Smoothed")

    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("CTC loss")
    plt.title("Test loss vs. training step")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_cer(run_dir, steps, cer_raw, cer_smooth=None, baseline=None):
    plt.figure(figsize=(6, 4))
    eps = 1e-5
    cer = np.maximum(cer_raw, eps)

    plt.plot(steps, cer, color=COL_PRIMARY, linewidth=1.8, label="Test PER")
    if cer_smooth is not None:
        plt.plot(steps, np.maximum(cer_smooth, eps), color=COL_SECONDARY,
                 linewidth=2.2, label="Smoothed")

    if baseline is not None:
        plt.axhline(max(baseline, eps), linestyle="--", color="dimgray",
                    label=f"Baseline = {baseline:.2f}")

    plt.yscale("log")        # CER log-scale: optional, but often helpful
    plt.xlabel("Training step")
    plt.ylabel("PER")
    plt.title("Test PER vs. training step")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "cer_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_joint(run_dir, steps, loss_raw, cer_raw, loss_smooth=None, cer_smooth=None):
    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    eps_l = 1e-8
    eps_c = 1e-5

    # --- Loss on left (log scale)
    loss = np.maximum(loss_raw, eps_l)
    if loss_smooth is not None:
        ax1.plot(steps, np.maximum(loss_smooth, eps_l), linewidth=2.2,
                 color=COL_SECONDARY, linestyle="--", label="Loss (smoothed)")
    else:
        ax1.plot(steps, loss, linewidth=1.8, linestyle="--", color=COL_SECONDARY, label="Loss")

    ax1.set_yscale("log")
    ax1.set_ylabel("CTC loss")
    ax1.set_xlabel("Training step")

    # --- CER on right (linear for clarity)
    cer_plot = cer_smooth if cer_smooth is not None else cer_raw
    ax2.plot(steps, cer_plot, linewidth=2.0, color=COL_PRIMARY,
             label="CER")
    ax2.set_ylabel("CER")

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    plt.title("Loss & CER vs. training step")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "loss_cer_joint.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# -------------------------------
# Main logic
# -------------------------------

def summarize_run(run_dir, log_every, smooth_window, baseline_cer=None):
    print("=" * 80)
    print(f"Run directory: {run_dir}")

    loss_raw, cer_raw = load_stats(run_dir)
    if loss_raw is None:
        return

    n_eval = len(loss_raw)
    steps = np.arange(1, n_eval + 1) * log_every

    print(f"  evals : {n_eval}")
    print(f"  final loss : {loss_raw[-1]:.6f}")
    print(f"  final CER  : {cer_raw[-1]:.6f}")
    print(f"  best CER   : {float(np.min(cer_raw)):.6f}")

    # smoothing
    loss_smooth = moving_average(loss_raw, smooth_window) if smooth_window > 1 else None
    cer_smooth = moving_average(cer_raw, smooth_window) if smooth_window > 1 else None

    # plot
    plot_loss(run_dir, steps, loss_raw, loss_smooth)
    plot_cer(run_dir, steps, cer_raw, cer_smooth, baseline=baseline_cer)
    plot_joint(run_dir, steps, loss_raw, cer_raw, loss_smooth, cer_smooth)

    # write summary
    summary_path = os.path.join(run_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"eval_points = {n_eval}\n")
        f.write(f"final_loss = {loss_raw[-1]:.6f}\n")
        f.write(f"final_CER  = {cer_raw[-1]:.6f}\n")
        f.write(f"best_CER   = {float(np.min(cer_raw)):.6f}\n")
    print(f"  summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-style training plots.")
    parser.add_argument("run_dirs", nargs="+",
                        help="Training run directories containing 'trainingStats'.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Batches between evals in training.")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Moving average window.")
    parser.add_argument("--baseline-cer", type=float, default=None,
                        help="Reference baseline CER (e.g. 0.28).")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            print(f"[WARN] {run_dir} is not a directory.")
            continue
        summarize_run(run_dir, args.log_every, args.smooth_window, args.baseline_cer)


if __name__ == "__main__":
    main()

