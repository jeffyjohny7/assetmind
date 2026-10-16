import numpy as np
from assetmind.models.anomaly_lstm import make_sequences, train_lstm, seq_errors


def test_make_sequences_shape_and_order():
    A = np.arange(50 * 3, dtype=float).reshape(50, 3)
    S = make_sequences(A, 30)
    assert S.shape == (21, 30, 3)
    assert np.allclose(S[0], A[:30]) and np.allclose(S[-1], A[-30:])


def test_autoencoder_rebuilds_normal_better_than_anomalies():
    t = np.arange(2000)
    normal = np.stack([np.sin(t / 5), np.cos(t / 5)], axis=1).astype(np.float32)   # a regular pattern
    seqs = make_sequences(normal, 30)
    model = train_lstm(seqs, hidden=16, epochs=10, batch=64, max_seqs=2000)

    normal_err = seq_errors(model, seqs[:200]).mean()
    noise = np.random.default_rng(0).normal(0, 1, (200, 30, 2)).astype(np.float32)   # never seen
    assert seq_errors(model, noise).mean() > 2 * normal_err