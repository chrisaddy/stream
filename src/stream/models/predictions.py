import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from stream.models.base import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_audit"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    input_hash: Mapped[str] = mapped_column(String(64))  # SHA256, not raw features
    risk_score: Mapped[float] = mapped_column(Float)
    risk_label: Mapped[str] = mapped_column(String(20))
    threshold_used: Mapped[float] = mapped_column(Float)
    top_shap_features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    llm_narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    inference_time_ms: Mapped[float] = mapped_column(Float)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewer_verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
