"""Synthetic KYC risk dataset generation.

We can't use real KYC data, but we demonstrate the pattern
with realistic synthetic data.
"""

import numpy as np
import pandas as pd


def generate_synthetic_kyc(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic KYC dataset.

    Features model what a regulated exchange would collect during onboarding.
    """
    rng = np.random.RandomState(seed)

    data = {
        "account_age_days": rng.exponential(180, n_samples).astype(int),
        "email_domain_type": rng.choice(
            ["free", "corporate", "disposable"], n_samples, p=[0.6, 0.3, 0.1]
        ),
        "phone_verified": rng.choice([0, 1], n_samples, p=[0.15, 0.85]),
        "document_type": rng.choice(
            ["passport", "drivers_license", "national_id"], n_samples, p=[0.4, 0.4, 0.2]
        ),
        "document_verification_score": np.clip(rng.normal(0.85, 0.15, n_samples), 0, 1),
        "ip_country_matches_document": rng.choice([0, 1], n_samples, p=[0.1, 0.9]),
        "device_fingerprint_seen_before": rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "transaction_velocity_first_24h": np.clip(rng.exponential(2, n_samples), 0, 50).astype(int),
        "referral_source": rng.choice(
            ["organic", "paid", "referral", "social"], n_samples, p=[0.4, 0.2, 0.25, 0.15]
        ),
        "signup_hour": rng.randint(0, 24, n_samples),
        "initial_deposit_usd": np.clip(rng.lognormal(5, 2, n_samples), 0, 100000),
    }

    df = pd.DataFrame(data)

    # Generate risk labels based on feature combinations
    risk_score = np.zeros(n_samples)
    risk_score += (df["email_domain_type"] == "disposable").astype(float) * 2
    risk_score += (1 - df["phone_verified"]) * 1.5
    risk_score += (1 - df["document_verification_score"]) * 3
    risk_score += (1 - df["ip_country_matches_document"]) * 2.5
    risk_score += (df["device_fingerprint_seen_before"]) * 1
    risk_score += np.clip(df["transaction_velocity_first_24h"] / 10, 0, 2)
    risk_score += (df["signup_hour"].isin([2, 3, 4])).astype(float) * 0.5
    risk_score += rng.normal(0, 0.5, n_samples)  # noise

    # Assign labels
    labels = pd.Series("low_risk", index=df.index)
    labels[risk_score > 3] = "medium_risk"
    labels[risk_score > 5] = "high_risk"
    labels[risk_score > 7] = "blocked"

    df["risk_label"] = labels

    return df


def encode_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode categorical features and return X, y."""
    df_encoded = df.copy()

    # Encode categoricals
    cat_cols = ["email_domain_type", "document_type", "referral_source"]
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    # Encode target
    label_map = {"low_risk": 0, "medium_risk": 1, "high_risk": 2, "blocked": 3}
    y = df_encoded["risk_label"].map(label_map).values
    X = df_encoded.drop("risk_label", axis=1).values.astype(float)

    return X, y
