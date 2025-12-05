import torch
from torch import nn

from .augmentations import GaussianSmoothing

def phones_to_diphones_batch(
    y: torch.Tensor,
    y_len: torch.Tensor,
    n_phones: int,
    sil_phone_id: int,
) -> torch.Tensor:
    """
    Convert phoneme sequences to diphone sequences (DCoND style).

    y:      (B, maxL) phoneme IDs, 0 = padding, 1..n_phones = real phones
    y_len:  (B,) lengths (# of non-zero phones per sequence)
    n_phones: number of phonemes (no blank), e.g. 41
    sil_phone_id: ID of SIL within [1..n_phones] (usually n_phones)

    Returns:
      y_diphone: (B, maxL) diphone IDs, 0 = padding, 1..(n_phones^2) = real diphones
         - Sequence length per row is same as y_len
    """
    device = y.device
    B, maxL = y.shape
    n_diphones = n_phones * n_phones

    y_diphone = torch.zeros_like(y, device=device)

    for b in range(B):
        L = int(y_len[b].item())
        if L <= 0:
            continue

        phones = y[b, :L]                 # (L,), values in 1..n_phones

        # prev phones: [SIL, p1, p2, ..., p_{L-1}]
        prev = torch.empty_like(phones)
        prev[0] = sil_phone_id
        if L > 1:
            prev[1:] = phones[:-1]

        # convert to zero-based indices: 0..n_phones-1
        prev0 = prev - 1
        curr0 = phones - 1

        # diphone index (prev, curr) -> 0..n_diphones-1
        diph0 = prev0 * n_phones + curr0

        # store as 1..n_diphones, 0 for padding beyond L
        y_diphone[b, :L] = diph0 + 1

    return y_diphone


class DCoNDGRUDecoderConvFrontend(nn.Module):
    """
    GRU + Conv frontend, trained on diphones, evaluated on phonemes.

    - Input:
        neuralInput: (B, T, neural_dim)
        dayIdx:      (B,) long

    - Output (forward):
        diphone_logits: (B, T', n_diphones + 1)
            where class 0 = diphone blank (CTC blank),
                  1..n_diphones = real diphones

    - marginalize_to_phonemes:
        takes diphone_logits (B, T', n_diphones+1)
        returns phoneme_probs (B, T', n_phones+1),
            where index 0 = phoneme blank,
                  1..n_phones = phonemes
    """

    def __init__(
        self,
        neural_dim: int,
        n_phones: int,       # number of phonemes (no blank), e.g. 41 (incl. SIL)
        hidden_dim: int,
        layer_dim: int,
        nDays: int = 24,
        dropout: float = 0.0,
        device: str = "cuda",
        strideLen: int = 4,
        kernelLen: int = 14,
        gaussianSmoothWidth: float = 0.0,
        bidirectional: bool = False,
        conv_dim: int = None,
    ):
        super().__init__()

        # --- hyperparameters ---
        self.neural_dim = neural_dim
        self.n_phones = n_phones          # P
        self.n_diphones = n_phones * n_phones   # D = P * P
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.nDays = nDays
        self.dropout = dropout
        self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        if conv_dim is None:
            conv_dim = neural_dim
        self.conv_dim = conv_dim

        self.inputLayerNonlinearity = nn.Softsign()

        # --- Gaussian smoothing over time ---
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim,
            20,
            self.gaussianSmoothWidth,
            dim=1,   # channel dimension
        )

        # --- per-day affine transform (neural_dim x neural_dim) ---
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        with torch.no_grad():
            for d in range(nDays):
                self.dayWeights[d].copy_(torch.eye(neural_dim))

        # --- temporal conv frontend ---
        # Use kernelLen + strideLen to match old Unfold-based downsampling.
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

        # --- GRU encoder ---
        rnn_input_dim = self.conv_dim
        self.gru_decoder = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # --- diphone output head ---
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # +1 for CTC blank
        self.fc_diphone_out = nn.Linear(out_dim, self.n_diphones + 1)

        # --- precompute diphone -> phoneme mapping for marginalization ---
        # diphone IDs: 0..n_diphones (0 = blank)
        # phoneme IDs: 0..n_phones  (0 = blank; 1..n_phones = real phones)
        # We build M of shape (n_diphones+1, n_phones+1), where:
        #   M[d, p] = 1 if diphone d has current phone p,
        #   d == 0 => blank diphone -> blank phoneme
        M = torch.zeros(self.n_diphones + 1, self.n_phones + 1)

        # map diphone blank -> phoneme blank
        M[0, 0] = 1.0

        # real diphones: 1..n_diphones
        for d in range(1, self.n_diphones + 1):
            d0 = d - 1
            curr_zero_based = d0 % self.n_phones      # 0..P-1
            curr_phone_id = curr_zero_based + 1       # 1..P
            M[d, curr_phone_id] = 1.0

        self.register_buffer("diphone_to_phone", M, persistent=False)

    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:
        """
        neuralInput: (B, T, neural_dim)
        dayIdx:      (B,)
        returns:
            diphone_logits: (B, T', n_diphones+1)
        """
        # --- Gaussian smoothing ---
        x = neuralInput.permute(0, 2, 1)       # (B, D, T)
        x = self.gaussianSmoother(x)
        x = x.permute(0, 2, 1)                 # (B, T, D)

        # --- per-day affine transform ---
        # dayWeights: (nDays, D, D) -> (B, D, D)
        W = torch.index_select(self.dayWeights, 0, dayIdx)   # (B, D, D)
        b = torch.index_select(self.dayBias, 0, dayIdx)      # (B, 1, D)

        # (B, T, D) x (B, D, D) -> (B, T, D)
        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.inputLayerNonlinearity(x)

        # --- conv frontend ---
        x = x.permute(0, 2, 1)                 # (B, D, T)
        x = self.conv_frontend(x)              # (B, conv_dim, T')
        x = x.permute(0, 2, 1)                 # (B, T', conv_dim)

        # --- GRU ---
        B = x.size(0)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.layer_dim * num_directions,
            B,
            self.hidden_dim,
            device=x.device,
        )
        hid, _ = self.gru_decoder(x, h0)       # (B, T', H*dir)

        diphone_logits = self.fc_diphone_out(hid)  # (B, T', n_diphones+1)
        return diphone_logits

    def marginalize_to_phonemes(self, diphone_logits: torch.Tensor) -> torch.Tensor:
        """
        DCoND-style marginalization:
          P(phone p | t) = sum_{diphones d with current_phone(d)=p} P(d | t)

        diphone_logits: (B, T', n_diphones+1), unnormalized
        returns:
          phoneme_probs: (B, T', n_phones+1)  [index 0 = blank]
        """
        diphone_probs = diphone_logits.softmax(dim=-1)        # (B, T', D+1)
        # (B, T', D+1) x (D+1, P+1) -> (B, T', P+1)
        M = self.diphone_to_phone.to(diphone_probs.device)
        phoneme_probs = torch.einsum("btd,dp->btp", diphone_probs, M)
        return phoneme_probs

