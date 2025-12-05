import torch
from torch.utils.data import Dataset
import numpy as np

class SpeechDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
    
    def compute_stats(self):
        feats = []

        for day in range(len(self.data)):
            for trial in range(len(self.data[day]["sentenceDat"])):
                x = self.data[day]["sentenceDat"][trial]  # shape (T, C), numpy
                x = np.log1p(np.clip(x, a_min=0, a_max=None))  # log transform
                feats.append(x)

        all_feats = np.concatenate(feats, axis=0)  # shape (sum_T, C)
        mean = all_feats.mean(axis=0)             # (C,)
        std = all_feats.std(axis=0)               # (C,)

        return torch.tensor(mean, dtype=torch.float32), \
            torch.tensor(std, dtype=torch.float32)
