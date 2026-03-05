"""Model race — 3 online classifiers competing in real-time."""

from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

_race = None


@dataclass
class RacingModel:
    name: str
    model: object
    rolling_f1: object = None
    rolling_accuracy: object = None
    cumulative_f1: object = None
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    n_correct: int = 0
    n_total: int = 0


class ModelRace:
    def __init__(self):
        from river.compose import Pipeline
        from river.linear_model import LogisticRegression
        from river.metrics import F1, Accuracy
        from river.naive_bayes import GaussianNB
        from river.preprocessing import StandardScaler
        from river.tree import HoeffdingTreeClassifier
        from river.utils import Rolling

        self.models = [
            RacingModel(
                name="LogisticRegression",
                model=Pipeline(StandardScaler(), LogisticRegression()),
                rolling_f1=Rolling(F1(), window_size=30),
                rolling_accuracy=Rolling(Accuracy(), window_size=30),
                cumulative_f1=F1(),
            ),
            RacingModel(
                name="HoeffdingTree",
                model=HoeffdingTreeClassifier(grace_period=10),
                rolling_f1=Rolling(F1(), window_size=30),
                rolling_accuracy=Rolling(Accuracy(), window_size=30),
                cumulative_f1=F1(),
            ),
            RacingModel(
                name="GaussianNB",
                model=GaussianNB(),
                rolling_f1=Rolling(F1(), window_size=30),
                rolling_accuracy=Rolling(Accuracy(), window_size=30),
                cumulative_f1=F1(),
            ),
        ]
        self.n_samples = 0

    def learn_one(self, x: dict, y: bool):
        """Test-then-train all models, track per-model metrics."""
        self.n_samples += 1

        for m in self.models:
            # Test
            y_pred = m.model.predict_one(x)
            if y_pred is not None:
                m.rolling_f1.update(y, y_pred)
                m.rolling_accuracy.update(y, y_pred)
                m.cumulative_f1.update(y, y_pred)
                m.n_total += 1
                if y_pred == y:
                    m.n_correct += 1

            # Train
            m.model.learn_one(x, y)

            # Record history
            try:
                f1_val = float(m.rolling_f1.get())
            except Exception:
                f1_val = 0.0
            m.history.append({"sample": self.n_samples, "f1": f1_val})

    def _ensemble_predict(self, x: dict) -> float:
        """F1-weighted majority vote across models."""
        total_weight = 0.0
        weighted_sum = 0.0

        for m in self.models:
            try:
                weight = max(float(m.rolling_f1.get()), 0.01)
            except Exception:
                weight = 0.01

            try:
                proba = m.model.predict_proba_one(x)
                p = proba.get(True, proba.get(1, 0.5))
            except Exception:
                p = 0.5

            weighted_sum += weight * p
            total_weight += weight

        return weighted_sum / max(total_weight, 1e-6)

    def get_standings(self) -> list[dict]:
        """Models sorted by rolling F1."""
        standings = []
        for m in self.models:
            try:
                rolling_f1 = float(m.rolling_f1.get())
            except Exception:
                rolling_f1 = 0.0
            try:
                rolling_acc = float(m.rolling_accuracy.get())
            except Exception:
                rolling_acc = 0.0
            try:
                cum_f1 = float(m.cumulative_f1.get())
            except Exception:
                cum_f1 = 0.0

            standings.append(
                {
                    "name": m.name,
                    "rolling_f1": round(rolling_f1, 4),
                    "rolling_accuracy": round(rolling_acc, 4),
                    "cumulative_f1": round(cum_f1, 4),
                    "n_total": m.n_total,
                }
            )

        standings.sort(key=lambda s: s["rolling_f1"], reverse=True)
        return standings

    def get_convergence_data(self) -> dict:
        """Per-model F1 history for Plotly chart."""
        data = {}
        for m in self.models:
            data[m.name] = list(m.history)
        return data

    def reset(self):
        """Reinitialize all models (called during adaptation)."""
        self.__init__()
        log.info("Model race reset")


def get_race() -> ModelRace:
    """Lazy-init singleton."""
    global _race
    if _race is None:
        _race = ModelRace()
    return _race


def reset_race():
    """Reset the singleton."""
    global _race
    _race = ModelRace()
    log.info("Model race singleton reset")
