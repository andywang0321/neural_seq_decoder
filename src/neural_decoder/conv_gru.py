import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoderConvFrontend(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        conv_dim=None,   # NEW: dimensionality of conv features before GRU
    ):
        super(GRUDecoderConvFrontend, self).__init__()

        # ----- hyperparams -----
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        # if conv_dim is not set, default to neural_dim
        if conv_dim is None:
            conv_dim = neural_dim
        self.conv_dim = conv_dim

        self.inputLayerNonlinearity = nn.Softsign()

        # Gaussian smoothing over time (same as before)
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # ----- per-day linear transform (same as before) -----
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # ----- temporal conv front-end (replaces Unfold) -----
        # Important: kernel_size = kernelLen, stride = strideLen, padding = 0
        # so the output time length matches the old Unfold-based length:
        #   L = floor((T - kernelLen)/strideLen) + 1
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(
                in_channels=neural_dim,
                out_channels=self.conv_dim,
                kernel_size=self.kernelLen,
                stride=self.strideLen,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
        )

        # ----- GRU layers (input size = conv_dim now) -----
        self.gru_decoder = nn.GRU(
            input_size=self.conv_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # weight init as before
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # ----- output projection -----
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1  # +1 for CTC blank
            )
        else:
            self.fc_decoder_out = nn.Linear(
                hidden_dim, n_classes + 1  # +1 for CTC blank
            )

    def forward(self, neuralInput, dayIdx):
        """
        neuralInput: (B, T, neural_dim)
        dayIdx:      (B,) long tensor of day indices
        returns:     (B, T', n_classes+1) where
                     T' = floor((T - kernelLen)/strideLen) + 1
        """

        # ----- Gaussian smoothing -----
        # (B, T, D) -> (B, D, T) -> smooth -> (B, T, D)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # ----- per-day linear transform -----
        # dayWeights: (B, D, D), dayBias: (B, 1, D)
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        dayBias = torch.index_select(self.dayBias, 0, dayIdx)  # (B, 1, D)

        # (B, T, D) @ (B, D, D) -> (B, T, D)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights)
        transformedNeural = transformedNeural + dayBias  # broadcast over T
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # ----- temporal conv front-end -----
        # (B, T, D) -> (B, D, T)
        x = transformedNeural.permute(0, 2, 1)
        # Conv1d: (B, D, T) -> (B, conv_dim, T')
        x = self.conv_frontend(x)
        # (B, conv_dim, T') -> (B, T', conv_dim)
        x = x.permute(0, 2, 1)

        # ----- GRU -----
        batch_size = transformedNeural.size(0)

        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                batch_size,
                self.hidden_dim,
                device=self.device,
            )
        else:
            h0 = torch.zeros(
                self.layer_dim,
                batch_size,
                self.hidden_dim,
                device=self.device,
            )

        # no need for requires_grad_ + detach; just use zeros
        hid, _ = self.gru_decoder(x, h0)

        # ----- output logits -----
        seq_out = self.fc_decoder_out(hid)  # (B, T', n_classes+1)
        return seq_out
