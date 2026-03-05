"""Prefect flow for illicit transaction detection training."""

import structlog
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from stream.illicit.data import load_pandas_dataset
from stream.illicit.evaluate import full_evaluation
from stream.illicit.preprocess import get_class_weight, temporal_split
from stream.illicit.xgboost_model import (
    get_global_feature_importance,
    get_shap_explainer,
    serialize_model,
    train_xgboost,
)
from stream.model_card import (
    build_model_card,
    confusion_matrix_chart,
    feature_importance_chart,
    fig_to_png,
    pr_curve_chart,
    save_model_card,
    temporal_chart,
)

log = structlog.get_logger()


@task(name="load-elliptic-dataset")
def load_dataset():
    log.info("Loading Elliptic Bitcoin dataset")
    features, labels, timesteps = load_pandas_dataset()
    log.info("Dataset loaded", n_samples=len(features), n_features=features.shape[1])
    return features, labels, timesteps


@task(name="split-data")
def split_data(features, labels, timesteps):
    X_train, y_train, X_test, y_test = temporal_split(features, labels, timesteps)
    log.info(
        "Temporal split complete",
        train_size=len(X_train),
        test_size=len(X_test),
        train_illicit=int(y_train.sum()),
        test_illicit=int(y_test.sum()),
    )
    return X_train, y_train, X_test, y_test


@task(name="train-xgboost-model")
def train_model(X_train, y_train, X_test, y_test):
    scale_pos_weight = get_class_weight(y_train)
    log.info("Training XGBoost", scale_pos_weight=f"{scale_pos_weight:.1f}")

    model = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        scale_pos_weight=scale_pos_weight,
    )
    log.info("XGBoost training complete", best_iteration=model.best_iteration)
    return model


@task(name="evaluate-model")
def evaluate_model(model, X_test, y_test, timesteps, labels):
    y_prob = model.predict_proba(X_test)[:, 1]

    # Get test timesteps
    known_mask = labels != 2
    test_mask = known_mask & (timesteps > 34)
    test_timesteps = timesteps[test_mask].values

    metrics = full_evaluation(y_test, y_prob, test_timesteps, model_name="xgboost")
    log.info(
        "Evaluation complete",
        pr_auc=f"{metrics['pr_auc']:.4f}",
        threshold=f"{metrics['cost_analysis']['threshold']:.3f}",
        precision=f"{metrics['cost_analysis']['precision']:.3f}",
        recall=f"{metrics['cost_analysis']['recall']:.3f}",
    )
    return metrics


@task(name="save-model-to-r2")
def save_model_to_r2(model, name: str = "illicit-xgboost"):
    """Save model to R2 via Prefect S3 block."""
    try:
        from io import BytesIO

        from prefect_aws.s3 import S3Bucket

        s3 = S3Bucket.load("model-store")
        model_bytes = serialize_model(model)
        s3.upload_from_file_object(BytesIO(model_bytes), f"models/{name}/latest.pkl")
        log.info("Model saved to R2", name=name, size_bytes=len(model_bytes))
    except Exception as e:
        # Fallback: save locally
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os

        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_model(model))
        log.info("Model saved locally", path=f"models/{name}/latest.pkl")


@task(name="create-metrics-artifact")
def create_artifact(metrics: dict):
    markdown = f"""# Illicit Detection — XGBoost Results

## Overall Metrics
| Metric | Value |
|--------|-------|
| PR-AUC | {metrics["pr_auc"]:.4f} |
| Samples | {metrics["n_samples"]:,} |
| Illicit | {metrics["n_illicit"]:,} |
| Class Ratio | {metrics["class_ratio"]} |

## Cost-Sensitive Threshold
| Metric | Value |
|--------|-------|
| Threshold | {metrics["cost_analysis"]["threshold"]:.3f} |
| Precision | {metrics["cost_analysis"]["precision"]:.3f} |
| Recall | {metrics["cost_analysis"]["recall"]:.3f} |
| F1 | {metrics["cost_analysis"]["f1"]:.3f} |
| False Negatives | {metrics["cost_analysis"]["fn_count"]} |
| False Positives | {metrics["cost_analysis"]["fp_count"]} |
| Total Cost | ${metrics["cost_analysis"]["total_cost"]:,.0f} |
"""
    create_markdown_artifact(key="illicit-xgboost-metrics", markdown=markdown)


@task(name="build-illicit-model-card")
def build_card(model, metrics, X_train, X_test, y_train, y_test):
    """Build model card with full metadata and save to R2."""

    # SHAP feature importance
    explainer = get_shap_explainer(model)
    fi = get_global_feature_importance(explainer, X_test, max_samples=500)

    # Confusion matrix from cost-sensitive threshold
    threshold = metrics["cost_analysis"]["threshold"]
    y_prob = model.predict_proba(X_test)[:, 1]
    preds = (y_prob >= threshold).astype(int)
    tp = int(((y_test == 1) & (preds == 1)).sum())
    fp = int(((y_test == 0) & (preds == 1)).sum())
    fn = int(((y_test == 1) & (preds == 0)).sum())
    tn = int(((y_test == 0) & (preds == 0)).sum())

    card = build_model_card(
        name="illicit-xgboost",
        description="XGBoost classifier for illicit Bitcoin transaction detection. "
        "Uses 166 transaction-level features from the Elliptic dataset.",
        metrics={
            "pr_auc": metrics["pr_auc"],
            "cost_analysis": metrics["cost_analysis"],
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        data_summary={
            "dataset": "Elliptic Bitcoin",
            "n_features": int(X_train.shape[1]),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_illicit": int(y_train.sum()),
            "test_illicit": int(y_test.sum()),
            "class_ratio": metrics["class_ratio"],
        },
        training_params={
            "model": "XGBClassifier",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "early_stopping_rounds": 20,
            "best_iteration": model.best_iteration,
        },
        feature_importance=fi[:20],
        curves={
            "pr_curve": metrics.get("pr_curve", {}),
            "temporal": metrics.get("temporal", []),
        },
    )

    save_model_card(card, "illicit-xgboost")
    log.info("Illicit model card saved")
    return card


@task(name="create-illicit-chart-artifacts")
def create_chart_artifacts(card: dict):
    """Create Prefect image artifacts for key charts."""
    try:
        from prefect.artifacts import create_image_artifact

        # PR curve
        pr = card["curves"].get("pr_curve", {})
        if pr.get("precision") and pr.get("recall"):
            fig = pr_curve_chart(pr["precision"], pr["recall"])
            create_image_artifact(fig_to_png(fig), key="illicit-pr-curve")

        # Feature importance
        fi = card.get("feature_importance", [])
        if fi:
            names = [f["feature"] for f in fi]
            scores = [f["importance"] for f in fi]
            fig = feature_importance_chart(names, scores)
            create_image_artifact(fig_to_png(fig), key="illicit-feature-importance")

        # Temporal stability
        temporal = card["curves"].get("temporal", [])
        if temporal:
            fig = temporal_chart(temporal)
            create_image_artifact(fig_to_png(fig), key="illicit-temporal-stability")

        # Confusion matrix
        cm = card["metrics"].get("confusion_matrix", {})
        if cm:
            fig = confusion_matrix_chart(cm["tp"], cm["fp"], cm["fn"], cm["tn"])
            create_image_artifact(fig_to_png(fig), key="illicit-confusion-matrix")

        log.info("Illicit chart artifacts created")
    except Exception as e:
        log.warning("Chart artifact creation failed", error=str(e))


@flow(name="illicit-detection-training")
def train_illicit_pipeline():
    """Full training pipeline: load → split → train → evaluate → save → card."""
    features, labels, timesteps = load_dataset()
    X_train, y_train, X_test, y_test = split_data(features, labels, timesteps)
    model = train_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test, timesteps, labels)
    save_model_to_r2(model)
    create_artifact(metrics)
    card = build_card(model, metrics, X_train, X_test, y_train, y_test)
    create_chart_artifacts(card)
    return metrics


if __name__ == "__main__":
    train_illicit_pipeline()
