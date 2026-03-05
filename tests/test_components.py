"""Unit tests for UI components."""

from stream.app.components import DemoBanner, FeedRow


class TestFeedRow:
    def test_respects_threshold_parameter(self):
        """FeedRow should use provided threshold, not hardcoded 0.7."""
        # Score of 0.5 should be "high" with threshold=0.4
        row = FeedRow("12:00:00", "abc123", 250, 10.0, 0.5, "HIGH", threshold=0.4)
        # Convert to string to check classes
        row_str = str(row)
        assert "high-risk" in row_str

    def test_low_score_not_high_risk(self):
        """Low score should not trigger high-risk class."""
        row = FeedRow("12:00:00", "abc123", 250, 10.0, 0.1, "LOW", threshold=0.7)
        row_str = str(row)
        assert "high-risk" not in row_str

    def test_model_name_badge_heuristic(self):
        """Should show HEURISTIC badge for heuristic model."""
        row = FeedRow(
            "12:00:00", "abc123", 250, 10.0, 0.5, "MED", threshold=0.7, model_name="live-heuristic"
        )
        row_str = str(row)
        assert "HEURISTIC" in row_str

    def test_model_name_badge_ml(self):
        """Should show ML badge for ML model."""
        row = FeedRow(
            "12:00:00", "abc123", 250, 10.0, 0.5, "MED", threshold=0.7, model_name="illicit-xgboost"
        )
        row_str = str(row)
        assert "ML" in row_str

    def test_no_badge_without_model_name(self):
        """Should not show any badge when model_name is empty."""
        row = FeedRow("12:00:00", "abc123", 250, 10.0, 0.5, "MED", threshold=0.7)
        row_str = str(row)
        assert "model-badge" not in row_str


class TestDemoBanner:
    def test_wraps_content(self):
        """DemoBanner should wrap its children in a demo-banner div."""
        banner = DemoBanner("Hello world")
        banner_str = str(banner)
        assert "demo-banner" in banner_str
        assert "Hello world" in banner_str

    def test_wraps_multiple_children(self):
        """DemoBanner should accept multiple children."""
        banner = DemoBanner("One", "Two", "Three")
        banner_str = str(banner)
        assert "demo-banner" in banner_str
