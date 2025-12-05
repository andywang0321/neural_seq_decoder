#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np


def load_args(run_dir):
    """Load the args pickle from a training run directory."""
    args_path = os.path.join(run_dir, "args")
    if not os.path.exists(args_path):
        print(f"[WARN] No args file in {run_dir}")
        return None
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    return args


def load_stats(run_dir):
    """Load the trainingStats pickle from a training run directory."""
    stats_path = os.path.join(run_dir, "trainingStats")
    if not os.path.exists(stats_path):
        print(f"[WARN] No trainingStats file in {run_dir}")
        return None
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    # handle both dict with np arrays or plain lists
    test_loss = np.array(stats.get("testLoss", []))
    test_cer = np.array(stats.get("testCER", []))
    return test_loss, test_cer


def summarize_run(run_dir, log_every=100):
    print("=" * 80)
    print(f"Run directory: {run_dir}")

    args = load_args(run_dir)
    if args is not None:
        # args may be a dict (hydra OmegaConf converted) or something similar
        print("\n[Config summary]")
        # Try to access like dict; if it's OmegaConf or similar, this still often works
        try:
            n_input = args.get("nInputFeatures", None)
            n_units = args.get("nUnits", None)
            n_layers = args.get("nLayers", None)
            kernel_len = args.get("kernelLen", None)
            stride_len = args.get("strideLen", None)
            bidir = args.get("bidirectional", None)
            lr_start = args.get("lrStart", None)
            lr_end = args.get("lrEnd", None)
            batch_size = args.get("batchSize", None)
            n_batch = args.get("nBatch", None)

            print(f"  nInputFeatures : {n_input}")
            print(f"  nUnits         : {n_units}")
            print(f"  nLayers        : {n_layers}")
            print(f"  kernelLen      : {kernel_len}")
            print(f"  strideLen      : {stride_len}")
            print(f"  bidirectional  : {bidir}")
            print(f"  batchSize      : {batch_size}")
            print(f"  nBatch         : {n_batch}")
            print(f"  lrStart        : {lr_start}")
            print(f"  lrEnd          : {lr_end}")
        except Exception as e:
            print(f"  [WARN] Could not pretty-print config: {e}")
    else:
        print("\n[Config summary] args not found.")

    stats = load_stats(run_dir)
    if stats is None:
        print("\n[Stats] trainingStats not found. Skipping.")
        return

    test_loss, test_cer = stats

    print("\n[Stats]")
    if len(test_loss) == 0 or len(test_cer) == 0:
        print("  No testLoss or testCER entries found.")
        return

    n_eval = len(test_cer)
    # evaluations happen every `log_every` batches in your trainer
    steps = np.arange(1, n_eval + 1) * log_every

    final_loss = float(test_loss[-1])
    final_cer = float(test_cer[-1])

    best_idx = int(np.argmin(test_cer))
    best_step = int(steps[best_idx])
    best_cer = float(test_cer[best_idx])
    best_loss = float(test_loss[best_idx])

    print(f"  # eval points      : {n_eval}")
    print(f"  Final step         : {int(steps[-1])}")
    print(f"  Final test loss    : {final_loss:.6f}")
    print(f"  Final test CER     : {final_cer:.6f}")
    print()
    print(f"  Best test CER      : {best_cer:.6f} at step {best_step}")
    print(f"  Corresponding loss : {best_loss:.6f}")

    # Optional: small table of first few and last few entries
    show_n = min(5, n_eval)
    print("\n  First few evals:")
    for i in range(show_n):
        print(
            f"    step {int(steps[i]):>6} | loss {float(test_loss[i]):>10.6f} "
            f"| CER {float(test_cer[i]):>8.6f}"
        )

    print("\n  Last few evals:")
    for i in range(max(0, n_eval - show_n), n_eval):
        print(
            f"    step {int(steps[i]):>6} | loss {float(test_loss[i]):>10.6f} "
            f"| CER {float(test_cer[i]):>8.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Read and summarize training log pickles (args, trainingStats)."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more training run directories containing 'args' and 'trainingStats'.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="How many batches between evals (default: 100, matching your trainer).",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            print(f"[WARN] {run_dir} is not a directory, skipping.")
            continue
        summarize_run(run_dir, log_every=args.log_every)


if __name__ == "__main__":
    main()

