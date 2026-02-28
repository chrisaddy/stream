"""FastHTML application — Stream Bitcoin ML Dashboard."""

import structlog
from fasthtml.common import *

from stream.app.models import load_all_models
from stream.app.pages.home import home_page
from stream.app.pages.illicit import illicit_page
from stream.app.pages.fees import fees_page
from stream.app.pages.lightning import lightning_page
from stream.app.pages.onboarding import onboarding_page
from stream.app.pages.alerts import alerts_page
from stream.app.pages.models_page import models_page
from stream.app.pages.pipeline import pipeline_page
from stream.app.pages.walkthrough import (
    walkthrough_overview,
    walkthrough_illicit,
    walkthrough_fees,
    walkthrough_architecture,
    walkthrough_integration,
    walkthrough_roadmap,
)
from stream.app.api import register_api_routes

log = structlog.get_logger()

# Create FastHTML app with static file serving
app, rt = fast_app(
    static_path="src/stream/app/static",
    hdrs=[
        Link(rel="stylesheet", href="/static/style.css"),
    ],
)

# Try to load models on startup (graceful fallback if not available)
try:
    load_all_models()
except Exception as e:
    log.warning("Model loading failed on startup", error=str(e))

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


@rt("/models")
def models():
    return models_page()


@rt("/pipeline")
def pipeline():
    return pipeline_page()


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


@rt("/walkthrough/integration")
def walkthrough_integ():
    return walkthrough_integration()


@rt("/walkthrough/roadmap")
def walkthrough_road():
    return walkthrough_roadmap()


# Run with: uvicorn stream.app.main:app --host 0.0.0.0 --port 5000
