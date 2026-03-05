import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from stream.models.base import Base


class AlertRecord(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tx_id: Mapped[str] = mapped_column(String(64))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_label: Mapped[str] = mapped_column(String(20))
    model_name: Mapped[str] = mapped_column(String(100))
    explanation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_input: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )  # {"vsize": int, "fee": int, "fee_rate": float}
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/reviewed/escalated
