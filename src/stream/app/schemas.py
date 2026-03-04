"""Pydantic request validation models."""

import re

from pydantic import BaseModel, field_validator


class IllicitScoreRequest(BaseModel):
    """Request body for the 166-feature illicit scoring endpoint."""

    features: str

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: str) -> str:
        parts = [x.strip() for x in v.split(",") if x.strip()]
        if len(parts) != 166:
            raise ValueError(f"Expected 166 comma-separated floats, got {len(parts)}")
        for i, p in enumerate(parts):
            try:
                float(p)
            except ValueError:
                raise ValueError(f"Feature at position {i} is not a valid number: {p!r}")
        return v


class LiveScoreRequest(BaseModel):
    """Request body for live transaction scoring (vsize + fee only)."""

    vsize: int
    fee: int

    @field_validator("vsize")
    @classmethod
    def validate_vsize(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vsize must be positive")
        return v

    @field_validator("fee")
    @classmethod
    def validate_fee(cls, v: int) -> int:
        if v < 0:
            raise ValueError("fee must be non-negative")
        return v


class LightningEvaluateRequest(BaseModel):
    """Request body for Lightning node evaluation by public key."""

    pubkey: str

    @field_validator("pubkey")
    @classmethod
    def validate_pubkey(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("pubkey is required")
        if not re.match(r"^[0-9a-fA-F]{66}$", v):
            raise ValueError("pubkey must be a 66-character hex string")
        return v


class ThresholdRequest(BaseModel):
    """Request body for updating the global risk threshold."""

    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        if not (0.05 <= v <= 0.9):
            raise ValueError("threshold must be between 0.05 and 0.9")
        return v
