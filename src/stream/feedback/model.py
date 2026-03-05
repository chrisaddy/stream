"""LightGBM model training on human feedback labels."""

import numpy as np
import structlog

log = structlog.get_logger()


def train_live_heuristic(X: np.ndarray, y: np.ndarray) -> dict:
    """Train a LightGBM binary classifier on analyst-labeled data.

    Uses cross-validation when <20 samples, train/test split otherwise.
    Returns dict with model object and evaluation metrics.
    """
    try:
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score, train_test_split
    except ImportError:
        log.error(
            "lightgbm or scikit-learn not installed"
            " — install with: pip install lightgbm scikit-learn"
        )
        raise

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "verbose": -1,
    }

    n_samples = len(y)
    n_positive = int(y.sum())
    n_negative = n_samples - n_positive
    log.info(
        "Training live heuristic", n_samples=n_samples, n_positive=n_positive, n_negative=n_negative
    )

    if n_samples < 20:
        # Cross-validation for tiny datasets
        model = lgb.LGBMClassifier(**params)
        k = min(5, n_samples)
        if k < 2 or n_positive < 2 or n_negative < 2:
            # Not enough diversity — just fit on everything
            model.fit(X, y)
            return {
                "model": model,
                "accuracy": float((model.predict(X) == y).mean()),
                "cv_scores": [],
                "n_train": n_samples,
                "n_test": 0,
                "method": "full_fit",
            }
        scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
        model.fit(X, y)
        return {
            "model": model,
            "accuracy": float(scores.mean()),
            "cv_scores": scores.tolist(),
            "n_train": n_samples,
            "n_test": 0,
            "method": f"{k}-fold_cv",
        }

    # Train/test split for larger datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    train_acc = float((model.predict(X_train) == y_train).mean())
    test_acc = float((model.predict(X_test) == y_test).mean())

    return {
        "model": model,
        "accuracy": test_acc,
        "train_accuracy": train_acc,
        "cv_scores": [],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "method": "train_test_split",
    }
