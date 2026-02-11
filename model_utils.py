import os
import pickle
from pathlib import Path


def save_model(model: dict, scaler=None, features=None, path: str = "models/model.pkl"):
    """Save model dict, scaler, and features to a pickle file."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "features": features,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_model(path: str = "models/model.pkl"):
    """Load saved model payload. Returns dict with keys `model`, `scaler`, `features`.
    Raises FileNotFoundError if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload
