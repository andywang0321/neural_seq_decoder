import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .conv_gru_diphone import DCoNDGRUDecoderConvFrontend, phones_to_diphones_batch
from .dataset import SpeechDataset


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


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    # Save args
    with open(os.path.join(args["outputDir"], "args"), "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    n_phones = args["nClasses"]      # e.g. 41
    sil_phone_id = n_phones          # SIL assumed to be last phoneme ID

    conv_dim = args.get("convDim", args["nInputFeatures"])

    model = DCoNDGRUDecoderConvFrontend(
        neural_dim=args["nInputFeatures"],
        n_phones=n_phones,
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

    # CTC over diphone classes: 0 = blank, 1..P^2 = diphone IDs
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
        total_iters=args["nBatch"],
    )

    testLoss = []
    testCER = []
    startTime = time.time()

    step = 0
    # We'll loop over the dataloader repeatedly until we hit nBatch updates
    while step < args["nBatch"]:
        for X, y, X_len, y_len, dayIdx in trainLoader:
            if step >= args["nBatch"]:
                break

            model.train()
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                dayIdx.to(device),
            )

            # Noise augmentation
            if args["whiteNoiseSD"] > 0:
                X = X + torch.randn_like(X) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X = X + (
                    torch.randn(X.shape[0], 1, X.shape[2], device=device)
                    * args["constantOffsetSD"]
                )

            # ----- build diphone targets -----
            y_diphone = phones_to_diphones_batch(
                y=y,
                y_len=y_len,
                n_phones=n_phones,
                sil_phone_id=sil_phone_id,
            )
            y_diphone_len = y_len  # same lengths as phonemes

            # ----- forward -----
            diphone_logits = model(X, dayIdx)

            # Length after conv/GRU; match baseline formula
            input_lengths = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            # CTCLoss expects (T, N, C)
            loss = loss_ctc(
                diphone_logits.log_softmax(2).permute(1, 0, 2),
                y_diphone,
                input_lengths,
                y_diphone_len,
            )
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            # --- Periodic evaluation ---
            if step % 100 == 0:
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

                        # Build diphone labels for loss (for monitoring)
                        y_val_diphone = phones_to_diphones_batch(
                            y=y_val,
                            y_len=y_len_val,
                            n_phones=n_phones,
                            sil_phone_id=sil_phone_id,
                        )
                        y_val_diphone_len = y_len_val

                        diphone_logits_val = model(X_val, dayIdx_val)
                        input_lengths_val = (
                            (X_len_val - model.kernelLen) / model.strideLen
                        ).to(torch.int32)

                        val_loss = loss_ctc(
                            diphone_logits_val.log_softmax(2).permute(1, 0, 2),
                            y_val_diphone,
                            input_lengths_val,
                            y_val_diphone_len,
                        )
                        val_loss = val_loss.sum()
                        allLoss.append(val_loss.cpu().numpy())

                        # ----- phoneme-level decoding via marginalization -----
                        phoneme_probs = model.marginalize_to_phonemes(diphone_logits_val)
                        # greedy phoneme decode
                        for i in range(phoneme_probs.shape[0]):
                            T_i = int(input_lengths_val[i].item())
                            frame_probs = phoneme_probs[i, :T_i, :]   # (T_i, P+1)

                            decoded = torch.argmax(frame_probs, dim=-1)  # (T_i,)
                            decoded = torch.unique_consecutive(decoded)
                            decoded = decoded[decoded != 0]             # drop blanks
                            decoded = decoded.cpu().numpy()

                            trueSeq = (
                                y_val[i, : y_len_val[i]]
                                .cpu()
                                .numpy()
                            )

                            matcher = SequenceMatcher(
                                a=trueSeq.tolist(), b=decoded.tolist()
                            )
                            total_edit_distance += matcher.distance()
                            total_seq_length += len(trueSeq)

                    avgDayLoss = float(np.sum(allLoss) / len(testLoader))
                    cer = total_edit_distance / total_seq_length

                    endTime = time.time()
                    print(
                        f"[DCoND] step {step}, "
                        f"ctc loss: {avgDayLoss:>7f}, "
                        f"phoneme CER: {cer:>7f}, "
                        f"time/100 steps: {(endTime - startTime):>7.3f} s"
                    )
                    startTime = time.time()

                # save model with best CER so far
                if len(testCER) == 0 or cer < min(testCER):
                    torch.save(
                        model.state_dict(),
                        os.path.join(args["outputDir"], "modelWeights"),
                    )
                testLoss.append(avgDayLoss)
                testCER.append(cer)

                tStats = {"testLoss": np.array(testLoss), "testCER": np.array(testCER)}
                with open(os.path.join(args["outputDir"], "trainingStats"), "wb") as f:
                    pickle.dump(tStats, f)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = os.path.join(modelDir, "modelWeights")
    with open(os.path.join(modelDir, "args"), "rb") as handle:
        args = pickle.load(handle)

    n_phones = args["nClasses"]
    conv_dim = args.get("convDim", args["nInputFeatures"])

    model = DCoNDGRUDecoderConvFrontend(
        neural_dim=args["nInputFeatures"],
        n_phones=n_phones,
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        conv_dim=conv_dim,
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)


if __name__ == "__main__":
    main()

