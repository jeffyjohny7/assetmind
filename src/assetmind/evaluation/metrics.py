import numpy as np


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric score from the C-MAPSS paper. Lower is better.
    Late predictions (overestimating remaining life) are penalized more,
    because the engine fails before maintenance is scheduled."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))