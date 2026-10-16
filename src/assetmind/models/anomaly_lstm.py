import numpy as np
import pandas as pd
import torch
from torch import nn
from numpy.lib.stride_tricks import sliding_window_view

SEQ_LEN = 30   # 30 one-minute windows = half an hour of context


def to_input(Z: pd.DataFrame, cols) -> np.ndarray:
    """Per-feature z-scores → model input: missing = 0 (at normal), log(1+z) to tame heavy tails."""
    return np.log1p(Z.reindex(columns=cols).fillna(0).to_numpy(dtype=np.float32))


def make_sequences(A: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """All consecutive sequences of length seq_len from one instance: shape (n - seq_len + 1, seq_len, features)."""
    if len(A) < seq_len:
        return np.empty((0, seq_len, A.shape[1]), dtype=np.float32)
    return sliding_window_view(A, (seq_len, A.shape[1]))[:, 0].astype(np.float32)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)                               # summary of the whole sequence
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)     # give the summary to every time step
        y, _ = self.decoder(repeated)
        return self.output(y)                                     # rebuilt sequence


def train_lstm(seqs: np.ndarray, hidden=32, epochs=8, batch=256, lr=1e-3, seed=42, max_seqs=20000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if len(seqs) > max_seqs:
        seqs = seqs[rng.choice(len(seqs), max_seqs, replace=False)]
    model = LSTMAutoencoder(seqs.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(seqs)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(batch):
            xb = X[idx]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def seq_errors(model, seqs: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Mean squared reconstruction error per sequence."""
    if len(seqs) == 0:
        return np.empty(0)
    out = []
    for i in range(0, len(seqs), batch):
        xb = torch.from_numpy(seqs[i:i + batch])
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(out)


def window_scores(model, A: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Score of each window = error of the sequence ending at it. First seq_len-1 windows: NaN."""
    scores = np.full(len(A), np.nan)
    if len(A) >= seq_len:
        scores[seq_len - 1:] = seq_errors(model, make_sequences(A, seq_len))
    return scores