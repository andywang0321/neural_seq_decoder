#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window):
    if window <= 1 or len(x) == 0:
        return x
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad to original length for easier plotting
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


def load_stats(run_dir):
    stats_path = os.path.join(run_dir, "trainingStats")
    if not os.path.exists(stats_path):
        print(f"[WARN] No trainingStats file found in {run_dir}, skipping.")
        return None, None

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    test_loss = np.array(stats.get("testLoss", []), dtype=float)
    test_cer = np.array(stats.get("testCER", []), dtype=float)

    if test_loss.size == 0 or test_cer.size == 0:
        print(f"[WARN] Empty testLoss/testCER in {run_dir}, skipping.")
        return None, None

    return test_loss, test_cer


def load_args(run_dir):
    args_path = os.path.join(run_dir, "args")
    if not os.path.exists(args_path):
        return None
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    # Try to handle OmegaConf or dict
    if hasattr(args, "keys") and not isinstance(args, dict):
        try:
            args = dict(args)
        except Exception:
            pass
    return args


def plot_loss(run_dir, steps, loss, loss_smooth=None):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, loss, label="Test CTC loss")
    if loss_smooth is not None:
        plt.plot(steps, loss_smooth, label="Smoothed loss", linewidth=2)

    plt.xlabel("Training step")
    plt.ylabel("CTC loss")
    plt.title("Test loss vs. training step")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def plot_cer(run_dir, steps, cer, cer_smooth=None, baseline=None):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, cer, label="Test CER")
    if cer_smooth is not None:
        plt.plot(steps, cer_smooth, label="Smoothed CER", linewidth=2)
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", label=f"Baseline CER = {baseline}")

    plt.xlabel("Training step")
    plt.ylabel("CER")
    plt.title("Test CER vs. training step")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(run_dir, "cer_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def plot_joint(run_dir, steps, loss, cer, loss_smooth=None, cer_smooth=None):
    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Loss on left axis
    if loss_smooth is not None:
        ax1.plot(steps, loss_smooth, label="Smoothed loss", linewidth=2)
    else:
        ax1.plot(steps, loss, label="Loss", linewidth=1)

    # CER on right axis
    if cer_smooth is not None:
        ax2.plot(steps, cer_smooth, label="Smoothed CER", linestyle="--")
    else:
        ax2.plot(steps, cer, label="CER", linestyle="--")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("CTC loss")
    ax2.set_ylabel("CER")

    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.title("Test loss and CER vs. training step")

    # Build a combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(run_dir, "loss_cer_joint.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def summarize_run(run_dir, log_every, smooth_window, baseline_cer=None):
    print("=" * 80)
    print(f"Run directory: {run_dir}")

    loss, cer = load_stats(run_dir)
    if loss is None or cer is None:
        return

    n_eval = len(loss)
    steps = np.arange(1, n_eval + 1) * log_every

    print(f"  # eval points : {n_eval}")
    print(f"  final step    : {int(steps[-1])}")
    print(f"  final loss    : {float(loss[-1]):.6f}")
    print(f"  final CER     : {float(cer[-1]):.6f}")
    print(f"  best CER      : {float(np.min(cer)):.6f}")

    # Optional smoothing
    loss_smooth = moving_average(loss, smooth_window) if smooth_window > 1 else None
    cer_smooth = moving_average(cer, smooth_window) if smooth_window > 1 else None

    # Create plots
    plot_loss(run_dir, steps, loss, loss_smooth)
    plot_cer(run_dir, steps, cer, cer_smooth, baseline=baseline_cer)
    plot_joint(run_dir, steps, loss, cer, loss_smooth, cer_smooth)

    # Optional: dump a small text summary
    summary_path = os.path.join(run_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"# eval points : {n_eval}\n")
        f.write(f"final step    : {int(steps[-1])}\n")
        f.write(f"final loss    : {float(loss[-1]):.6f}\n")
        f.write(f"final CER     : {float(cer[-1]):.6f}\n")
        f.write(f"best CER      : {float(np.min(cer)):.6f}\n")
    print(f"  summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate professional-looking training curves from Neural Speech logs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more training run directories containing 'trainingStats'.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Number of batches between evals (matches the trainer; default: 100).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving average window for smoothing curves (default: 5; set to 1 to disable).",
    )
    parser.add_argument(
        "--baseline-cer",
        type=float,
        default=None,
        help="Optional baseline CER to plot as a horizontal reference (e.g., 0.28).",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            print(f"[WARN] {run_dir} is not a directory, skipping.")
            continue
        summarize_run(
            run_dir,
            log_every=args.log_every,
            smooth_window=args.smooth_window,
            baseline_cer=args.baseline_cer,
        )


if __name__ == "__main__":
    main()

