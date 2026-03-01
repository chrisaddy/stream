import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from stream.models.base import Base


class ReviewRecord(Base):
    __tablename__ = "reviews"

    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    alert_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    tx_id: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(20))  # true_positive / false_positive
    reviewer: Mapped[str] = mapped_column(String(100), default="analyst")
    risk_score_at_review: Mapped[float] = mapped_column(Float)
    threshold_at_review: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
