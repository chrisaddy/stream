"""Model loading from R2 or local storage."""

import os
import pickle
import structlog

log = structlog.get_logger()

# Module-level model cache
_models: dict = {}


def _load_from_r2(path: str) -> bytes | None:
    """Try to load model bytes from R2 via Prefect S3 block."""
    try:
        import asyncio
        from prefect_aws.s3 import S3Bucket
        s3 = asyncio.get_event_loop().run_until_complete(S3Bucket.load("model-store"))
        return s3.read_path(path)
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
