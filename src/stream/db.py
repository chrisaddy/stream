import asyncio

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from stream.config import settings

log = structlog.get_logger()

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Async-safe DB write helpers ---


async def write_in_thread(func, *args, **kwargs):
    """Run a sync DB function in a thread so it doesn't block the event loop."""
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as e:
        log.warning("DB write failed", func=func.__name__, error=str(e))
        return None


def save_prediction(kwargs: dict):
    """Create a PredictionRecord row. Runs in a thread via write_in_thread."""
    from stream.models.predictions import PredictionRecord

    db = SessionLocal()
    try:
        record = PredictionRecord(**kwargs)
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        log.warning("save_prediction failed", error=str(e))
    finally:
        db.close()


def save_alert(kwargs: dict):
    """Create an AlertRecord row. Runs in a thread via write_in_thread."""
    from stream.models.alerts import AlertRecord

    db = SessionLocal()
    try:
        record = AlertRecord(**kwargs)
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        log.warning("save_alert failed", error=str(e))
    finally:
        db.close()


def save_review(kwargs: dict):
    """Create a ReviewRecord row. Runs in a thread via write_in_thread."""
    from stream.models.reviews import ReviewRecord

    db = SessionLocal()
    try:
        record = ReviewRecord(**kwargs)
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        log.warning("save_review failed", error=str(e))
    finally:
        db.close()
