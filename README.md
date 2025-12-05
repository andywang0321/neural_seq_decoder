<img src="media/campus-seal.jpg" alt="UCLA logo" width="100" height="100">

# neural-seq-decoder

Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder) for UCLA EEC143A Final Project

## Installation

Use the package manager [uv](https://docs.astral.sh/uv/) to install neural-seq-decoder.

```bash
uv sync
```

## Usage

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)

2. Train model

```python
uv run scripts/train_model.py
```
## Records

starting off with shreeram's hyperparameters, the baseline line model had a PER of 0.286616.

First we try changing the optimizer from Adam to AdamW and reducing the epsilon to 1e-8. These two changes made the steps too large and the model is not able to optimize.
Then the epsilon was lowered to 0.01. This still did not work and model diverged

Just changing the optimizer to AdamW with original parameters did not decrease the error rate batch 2900, ctc loss: 0.995849, cer: 0.286286, time/batch:   0.159

tried time-masking in ebrahim's paper



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
