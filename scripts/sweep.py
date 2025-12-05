import os
from copy import deepcopy
import time
import wandb
from neural_decoder.neural_decoder_trainer import trainModel

# -------- base args (your current config) ----------
BASE_MODEL_NAME = "speechBaseline5"
BASE_OUTPUT_DIR = "/home/matthewli/data/brain2speech/logs/speech_logs/"
BASE_DATASET_PATH = "/home/matthewli/data/brain2speech/ptDecoder_ctc"

BASE_ARGS = {
    "outputDir": None,  # will be set per run
    "datasetPath": BASE_DATASET_PATH,
    "seqLen": 150,
    "maxTimeSeriesLen": 1200,
    "batchSize": 128,
    "lrStart": 0.05,
    "lrEnd": 0.02,
    "nUnits": 256,
    "nBatch": 3000,
    "nLayers": 5,
    "seed": 0,
    "nClasses": 40,
    "nInputFeatures": 256,
    "dropout": 0.2,
    "whiteNoiseSD": 0.8,
    "constantOffsetSD": 0.2,
    "gaussianSmoothWidth": 2.0,
    "strideLen": 4,
    "kernelLen": 32,
    "bidirectional": False,
    "l2_decay": 1e-5,
    "eps": 0.1,
    "timeMask_maxWidth" :0,
    "timeMask_nMasks":0,
    "timeMask_p":0
}

# -------- W&B sweep config ------------
sweep_config = {
    "method": "random",  # or "grid" if you prefer
    "metric": {
        "name": "val/cer",   # make sure you log this in trainModel
        "goal": "minimize",
    },
    "parameters": {
        "lrStart": {
            "values": [0.008, 0.012, 0.015, 0.018, 0.022]
        },
        "lrEnd": {
            "values": [1e-5, 3e-5, 1e-4]
        },
        "l2_decay": {
            "values": [3e-5, 1e-4, 3e-4, 1e-3]
        },
        "eps": {
            "values": [1e-4, 1e-3, 1e-2]
        }
    }

}



def sweep_train(config=None):
    with wandb.init(config=config, project="neural_decoder", entity="ali-matthew"):
        cfg = wandb.config

        args = deepcopy(BASE_ARGS)
        args["lrStart"] = cfg.lrStart
        # keep same lrEnd/lrStart ratio
        args["lrEnd"] = cfg.lrStart * (BASE_ARGS["lrEnd"] / BASE_ARGS["lrStart"])
        args["l2_decay"] = cfg.l2_decay
        args["eps"] = cfg.eps
        args["outputDir"] = f"{BASE_OUTPUT_DIR}_{time.time}"
        
        args["lrStart"] = cfg.lrStart
        args["lrEnd"] = cfg.lrEnd

        run_name = (
            f"{BASE_MODEL_NAME}"
            f"_lr{args['lrStart']}_wd{args['l2_decay']}_eps{args['eps']}"
        )

        wandb.run.name = run_name

        print("\n================ RUN CONFIG ================")
        print("lrStart:", args["lrStart"], "lrEnd:", args["lrEnd"])
        print("l2_decay:", args["l2_decay"])
        print("eps:", args["eps"])
        print("==========================================\n")

        trainModel(args)


if __name__ == "__main__":
    # Create sweep on W&B
    sweep_id = wandb.sweep(
    sweep_config,
    project="neural_decoder",  # or any name you want
    entity="ali-matthew",   
)


    # Launch N runs from this machine
    wandb.agent(sweep_id, function=sweep_train, count=20)
