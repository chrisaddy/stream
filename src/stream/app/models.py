"""Model loading from R2 or local storage."""

import os
import pickle
import structlog

log = structlog.get_logger()

# Module-level model cache
_models: dict = {}
_model_cards: dict = {}


def _load_from_r2(path: str) -> bytes | None:
    """Try to load model bytes from R2 via boto3 (sync-safe)."""
    try:
        import boto3
        from stream.config import settings
        if not settings.R2_ENDPOINT_URL:
            return None
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        )
        resp = s3.get_object(Bucket=settings.R2_BUCKET_NAME, Key=path)
        return resp["Body"].read()
    except Exception as e:
        log.debug("R2 load failed", path=path, error=str(e))
        return None


def _load_from_local(path: str) -> bytes | None:
    """Try to load model bytes from local file."""
    local_path = path
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    return None


def load_model(name: str, deserializer=None):
    """Load a model by name. Tries R2 first, then local."""
    if name in _models:
        return _models[name]

    path = f"models/{name}/latest.pkl"

    data = _load_from_r2(path) or _load_from_local(path)
    if data is None:
        log.warning("Model not found", name=name)
        return None

    if deserializer:
        model = deserializer(data)
    else:
        from io import BytesIO
        model = pickle.load(BytesIO(data))

    _models[name] = model
    log.info("Model loaded", name=name)
    return model


def get_model(name: str):
    """Get a cached model."""
    return _models.get(name)


def reload_model(name: str, deserializer=None):
    """Force reload a model."""
    _models.pop(name, None)
    return load_model(name, deserializer)


def load_all_models():
    """Load all models on startup."""
    model_names = ["illicit-xgboost", "fee-lgbm", "lightning-lgbm", "onboarding-xgb"]
    for name in model_names:
        load_model(name)
    log.info("All models loaded", count=len(_models))


def load_model_cards():
    """Load all model card JSONs from R2 on startup."""
    from stream.model_card import load_model_card_from_r2

    card_names = ["illicit-xgboost", "fee-lgbm", "lightning-lgbm", "onboarding-xgb"]
    for name in card_names:
        card = load_model_card_from_r2(name)
        if card:
            _model_cards[name] = card
            log.info("Model card loaded", name=name)
        else:
            log.debug("No model card found", name=name)
    log.info("Model cards loaded", count=len(_model_cards))


def get_model_card(name: str) -> dict | None:
    """Get a cached model card."""
    return _model_cards.get(name)


def get_all_model_cards() -> dict:
    """Get all cached model cards."""
    return _model_cards
