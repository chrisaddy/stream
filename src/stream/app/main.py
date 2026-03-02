"""FastHTML application — Stream Bitcoin ML Dashboard."""

import structlog
from fasthtml.common import *

from stream.app.models import load_all_models, load_lightning_features, load_model_cards
from stream.db import engine
from stream.models.base import Base
from stream.models.predictions import PredictionRecord  # noqa: F401 — register table
from stream.models.alerts import AlertRecord  # noqa: F401 — register table
from stream.models.reviews import ReviewRecord  # noqa: F401 — register table
from stream.models.settings import Setting  # noqa: F401 — register table
from stream.app.pages.home import home_page
from stream.app.pages.illicit import illicit_page
from stream.app.pages.fees import fees_page
from stream.app.pages.lightning import lightning_page
from stream.app.pages.onboarding import onboarding_page
from stream.app.pages.alerts import alerts_page
from stream.app.pages.walkthrough import (
    walkthrough_overview,
    walkthrough_illicit,
    walkthrough_fees,
    walkthrough_architecture,
    walkthrough_online_ml,
    walkthrough_compliance,
)
from stream.app.api import register_api_routes

log = structlog.get_logger()


async def on_startup():
    # 1. Create DB tables (idempotent)
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Database tables ensured")
    except Exception as e:
        log.warning("Database table creation failed", error=str(e))

    # 2. Load ML models
    try:
        load_all_models()
    except Exception as e:
        log.warning("Model loading failed on startup", error=str(e))

    # 3. Load model cards
    try:
        load_model_cards()
    except Exception as e:
        log.warning("Model card loading failed on startup", error=str(e))

    # 4. Load lightning features cache
    try:
        load_lightning_features()
    except Exception as e:
        log.warning("Lightning features loading failed on startup", error=str(e))

    # 5. Warm up online model from existing reviews
    try:
        from stream.feedback.pipeline import warm_up_river_model
        warm_up_river_model()
    except Exception as e:
        log.warning("Online ML warm-up failed on startup", error=str(e))


# Create FastHTML app with static file serving
app, rt = fast_app(
    static_path="src/stream/app/static",
    on_startup=[on_startup],
)

# Register API routes
register_api_routes(rt)


# === PAGE ROUTES ===

@rt("/")
def index():
    return home_page()


@rt("/illicit")
def illicit():
    return illicit_page()


@rt("/fees")
def fees():
    return fees_page()


@rt("/lightning")
def lightning():
    return lightning_page()


@rt("/onboarding")
def onboarding():
    return onboarding_page()


@rt("/alerts")
def alerts():
    return alerts_page()


@rt("/walkthrough")
def walkthrough():
    return walkthrough_overview()


@rt("/walkthrough/illicit")
def walkthrough_illicit_page():
    return walkthrough_illicit()


@rt("/walkthrough/fees")
def walkthrough_fees_page():
    return walkthrough_fees()


@rt("/walkthrough/architecture")
def walkthrough_arch():
    return walkthrough_architecture()


@rt("/walkthrough/online-ml")
def walkthrough_online_ml_page():
    return walkthrough_online_ml()


@rt("/walkthrough/compliance")
def walkthrough_compliance_page():
    return walkthrough_compliance()


# Run with: uvicorn stream.app.main:app --host 0.0.0.0 --port 5000
