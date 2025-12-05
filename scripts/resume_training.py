#!/usr/bin/env python3
import os
import argparse
import pickle
import time

import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.conv_gru import GRUDecoderConvFrontend  # adjust import path if needed
from neural_decoder.dataset import SpeechDataset         # adjust import path if needed


def getDatasetLoaders(datasetName, batchSize):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def resume_training(run_dir, extra_batches, output_dir=None, log_every=100, device="cuda"):
    # --- load previous args ---
    args_path = os.path.join(run_dir, "args")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"No args file found in {run_dir}")
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    # convert OmegaConf to dict if needed (Hydra), otherwise assume dict-like
    if hasattr(args, "keys") and not isinstance(args, dict):
        # OmegaConf-like; convert to dict
        args = dict(args)

    # where to save resumed stuff
    if output_dir is None:
        output_dir = run_dir  # continue in-place
    os.makedirs(output_dir, exist_ok=True)

    # save a copy of (possibly updated) args
    args["outputDir"] = output_dir
    with open(os.path.join(output_dir, "args"), "wb") as f:
        pickle.dump(args, f)

    # --- dataset ---
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    # --- build model ---
    conv_dim = args.get("convDim", args["nInputFeatures"])

    model = GRUDecoderConvFrontend(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        conv_dim=conv_dim,
    ).to(device)

    # --- load previous weights ---
    weight_path = os.path.join(run_dir, "modelWeights")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"No modelWeights file found in {run_dir}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {weight_path}")

    # --- CTC loss, optimizer, scheduler ---
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=extra_batches,
    )

    # --- load previous stats (if present) ---
    stats_path = os.path.join(run_dir, "trainingStats")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        testLoss = list(stats.get("testLoss", []))
        testCER = list(stats.get("testCER", []))
        # infer where we left off (approximate global step)
        prev_eval_count = len(testCER)
        start_step = prev_eval_count * log_every
        best_cer_so_far = min(testCER) if len(testCER) > 0 else float("inf")
        print(f"Loaded previous stats: {prev_eval_count} evals, best CER {best_cer_so_far:.6f}")
    else:
        testLoss = []
        testCER = []
        start_step = 0
        best_cer_so_far = float("inf")
        print("No previous trainingStats found; starting fresh stats.")

    # --- training loop: continue from start_step for extra_batches ---
    model.train()
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    total_batches = start_step + extra_batches
    current_step = start_step
    start_time = time.time()

    print(
        f"Resuming training from step {start_step}, "
        f"running for {extra_batches} more steps (to {total_batches})."
    )

    # We'll iterate the dataloader in a loop until we hit total_batches
    train_iter = iter(trainLoader)

    while current_step < total_batches:
        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # noise augmentation
        if args["whiteNoiseSD"] > 0:
            X = X + torch.randn_like(X) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X = X + (
                torch.randn(X.shape[0], 1, X.shape[2], device=device)
                * args["constantOffsetSD"]
            )

        # forward
        pred = model(X, dayIdx)

        # compute input lengths after conv frontend
        # (using the same formula as original training)
        input_lengths = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        loss = loss_ctc(
            pred.log_softmax(2).permute(1, 0, 2),
            y,
            input_lengths,
            y_len,
        )
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_step += 1

        # --- evaluation ---
        if current_step % log_every == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0

                for X_val, y_val, X_len_val, y_len_val, dayIdx_val in testLoader:
                    X_val, y_val, X_len_val, y_len_val, dayIdx_val = (
                        X_val.to(device),
                        y_val.to(device),
                        X_len_val.to(device),
                        y_len_val.to(device),
                        dayIdx_val.to(device),
                    )

                    pred_val = model(X_val, dayIdx_val)
                    input_lengths_val = (
                        (X_len_val - model.kernelLen) / model.strideLen
                    ).to(torch.int32)

                    val_loss = loss_ctc(
                        pred_val.log_softmax(2).permute(1, 0, 2),
                        y_val,
                        input_lengths_val,
                        y_len_val,
                    )
                    val_loss = val_loss.sum()
                    allLoss.append(val_loss.cpu().item())

                    # CER computation
                    for i in range(pred_val.shape[0]):
                        T_i = int(input_lengths_val[i].item())
                        decodedSeq = torch.argmax(
                            pred_val[i, :T_i, :], dim=-1
                        )  # (T_i,)
                        decodedSeq = torch.unique_consecutive(decodedSeq)
                        decodedSeq = decodedSeq[decodedSeq != 0]  # drop blanks
                        decodedSeq = decodedSeq.cpu().numpy()

                        trueSeq = (
                            y_val[i, : y_len_val[i]]
                            .cpu()
                            .numpy()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avg_loss = float(np.mean(allLoss))
                cer = total_edit_distance / total_seq_length

                end_time = time.time()
                print(
                    f"[resume] step {current_step}, "
                    f"ctc loss: {avg_loss:>7f}, "
                    f"cer: {cer:>7f}, "
                    f"time/{log_every} steps: {(end_time - start_time):>7.3f}s"
                )
                start_time = time.time()

                # update best CER and save weights if improved
                if cer < best_cer_so_far:
                    best_cer_so_far = cer
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "modelWeights"),
                    )
                    print(f"  New best CER {best_cer_so_far:.6f}, weights saved.")

                # append stats
                testLoss.append(avg_loss)
                testCER.append(cer)

                # save stats
                with open(os.path.join(output_dir, "trainingStats"), "wb") as f:
                    pickle.dump(
                        {
                            "testLoss": np.array(testLoss),
                            "testCER": np.array(testCER),
                        },
                        f,
                    )

            model.train()

    print("Resume training finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Resume training of a GRU+Conv model from a previous run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory of the previous run (must contain 'args' and 'modelWeights').",
    )
    parser.add_argument(
        "--extra-batches",
        type=int,
        required=True,
        help="How many additional training steps (batches) to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save resumed outputs. "
             "Defaults to --run-dir (continue in-place).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="How many steps between evals (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    args = parser.parse_args()

    resume_training(
        run_dir=args.run_dir,
        extra_batches=args.extra_batches,
        output_dir=args.output_dir,
        log_every=args.log_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()

