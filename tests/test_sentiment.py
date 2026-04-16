"""Tests for NLP sentiment features (Jansen)."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.sentiment import (
    FinBERTSentimentModel,
    NewsFetcher,
    aggregate_sentiment,
    compute_sentiment_features,
    sentiment_momentum,
)


# ---------------------------------------------------------------------------
# Mock scorer used by tests that need a deterministic sentiment_model
# ---------------------------------------------------------------------------

class MockSentimentModel:
    """
    Deterministic sentiment scorer. Maps keywords in the text to fixed scores
    so tests can exercise the aggregation + feature pipeline without
    downloading the real FinBERT model.
    """

    def __init__(self, default_score: float = 0.0) -> None:
        self.default_score = default_score
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[dict]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            tl = t.lower()
            if "surge" in tl or "beat" in tl or "rally" in tl:
                out.append({"positive": 0.8, "neutral": 0.15, "negative": 0.05, "sentiment_score": 0.75})
            elif "plunge" in tl or "miss" in tl or "crash" in tl:
                out.append({"positive": 0.05, "neutral": 0.15, "negative": 0.80, "sentiment_score": -0.75})
            else:
                s = self.default_score
                out.append({"positive": 0.0, "neutral": 1.0, "negative": 0.0, "sentiment_score": s})
        return out


# ---------------------------------------------------------------------------
# FinBERTSentimentModel — tested without downloading the real model
# ---------------------------------------------------------------------------

class TestFinBERTSentimentModel:
    def test_construction_does_not_download(self):
        """The constructor must not touch the network / disk model cache."""
        # Simply constructing should not raise even if transformers is absent.
        m = FinBERTSentimentModel(device="cpu", batch_size=8)
        assert m._pipeline is None
        assert m.batch_size == 8

    def test_empty_predict_returns_empty(self):
        m = FinBERTSentimentModel()
        assert m.predict([]) == []
        # Still no model loaded.
        assert m._pipeline is None

    def test_predict_with_stubbed_pipeline(self, monkeypatch):
        """Stub the HuggingFace pipeline output and verify our post-processing."""
        m = FinBERTSentimentModel()
        # Emulate the top_k=None pipeline output: list of per-input label dicts.
        stub_output = [
            [
                {"label": "positive", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "negative", "score": 0.1},
            ],
            [
                {"label": "positive", "score": 0.05},
                {"label": "neutral", "score": 0.10},
                {"label": "negative", "score": 0.85},
            ],
        ]

        def fake_pipeline(texts, batch_size=None):
            assert texts == ["a", "b"]
            return stub_output

        m._pipeline = fake_pipeline  # short-circuit lazy load
        out = m.predict(["a", "b"])
        assert len(out) == 2
        np.testing.assert_allclose(out[0]["positive"], 0.7)
        np.testing.assert_allclose(out[0]["sentiment_score"], 0.6)  # 0.7 - 0.1
        np.testing.assert_allclose(out[1]["sentiment_score"], -0.80)

    def test_predict_raises_if_transformers_missing(self, monkeypatch):
        """If transformers isn't installed the lazy loader must surface the error."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        m = FinBERTSentimentModel()
        with pytest.raises(ImportError):
            m.predict(["hello"])


# ---------------------------------------------------------------------------
# aggregate_sentiment
# ---------------------------------------------------------------------------

class TestAggregateSentiment:
    def test_empty_input_returns_zero(self):
        assert aggregate_sentiment([], []) == 0.0

    def test_recent_positive_dominates_old_negative(self):
        """A fresh +1 should dominate a 48h-old -1 with half-life of 24h."""
        now = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
        scores = [
            {"sentiment_score": -1.0},  # old & negative
            {"sentiment_score": 1.0},   # new & positive
        ]
        timestamps = [now - timedelta(hours=48), now]
        agg = aggregate_sentiment(scores, timestamps, half_life_hours=24.0)
        # Weights: exp(-48/24)=0.135, exp(0)=1.0. Aggregate = (−0.135 + 1) / 1.135 ≈ 0.762
        assert agg > 0.5
        assert abs(agg - 0.762) < 0.02

    def test_single_score(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        agg = aggregate_sentiment(
            [{"sentiment_score": 0.3}], [now], half_life_hours=12.0
        )
        assert agg == pytest.approx(0.3)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            aggregate_sentiment(
                [{"sentiment_score": 0.1}], [datetime.now(), datetime.now()]
            )

    def test_invalid_half_life(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            aggregate_sentiment(
                [{"sentiment_score": 0.0}], [now], half_life_hours=-1.0
            )


# ---------------------------------------------------------------------------
# sentiment_momentum
# ---------------------------------------------------------------------------

class TestSentimentMomentum:
    def test_returns_expected_columns_and_differences(self):
        s = pd.Series([0.1, 0.2, 0.25, 0.5, 0.4, 0.3])
        df = sentiment_momentum(s, windows=[1, 3])
        assert list(df.columns) == ["sentiment_mom_1", "sentiment_mom_3"]
        # mom_1 is diff(1): NaN, 0.1, 0.05, 0.25, -0.1, -0.1
        np.testing.assert_allclose(
            df["sentiment_mom_1"].iloc[1:].values, [0.1, 0.05, 0.25, -0.1, -0.1]
        )
        # mom_3 at t=3: 0.5 - 0.1 = 0.4
        np.testing.assert_allclose(df["sentiment_mom_3"].iloc[3], 0.4)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            sentiment_momentum(pd.Series([0.1]), windows=[0])


# ---------------------------------------------------------------------------
# NewsFetcher — HTTP mocked via monkeypatched _http_get
# ---------------------------------------------------------------------------

def _fake_newsapi_response(articles: list[dict]) -> str:
    payload = {"status": "ok", "totalResults": len(articles), "articles": articles}
    return json.dumps(payload)


class TestNewsFetcher:
    def test_fetch_returns_parsed_articles(self, monkeypatch):
        fake_articles = [
            {
                "title": "AAPL surges on strong earnings",
                "description": "The tech giant beat expectations.",
                "publishedAt": "2024-03-10T14:30:00Z",
                "source": {"name": "Example Wire"},
                "url": "https://example.com/aapl",
            },
            {
                "title": "Market sentiment improves",
                "description": None,
                "publishedAt": "2024-03-10T12:00:00Z",
                "source": {"name": "Other News"},
                "url": "https://example.com/mkt",
            },
        ]
        fetcher = NewsFetcher(api_key="dummy")
        monkeypatch.setattr(
            fetcher, "_http_get", lambda url, timeout=10: _fake_newsapi_response(fake_articles)
        )

        start = datetime(2024, 3, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 3, 11, 0, 0, tzinfo=timezone.utc)
        out = fetcher.fetch("AAPL", start=start, end=end)

        assert len(out) == 2
        assert out[0]["title"] == "AAPL surges on strong earnings"
        assert out[0]["source"] == "Example Wire"
        # Timestamp parsed into timezone-aware datetime.
        assert out[0]["timestamp"].tzinfo is not None
        # Missing description becomes empty string, not None.
        assert out[1]["description"] == ""

    def test_fetch_handles_http_error(self, monkeypatch):
        fetcher = NewsFetcher(api_key="dummy")

        def boom(url, timeout=10):
            raise ConnectionError("network down")

        monkeypatch.setattr(fetcher, "_http_get", boom)
        out = fetcher.fetch(
            "AAPL",
            start=datetime(2024, 3, 10, tzinfo=timezone.utc),
            end=datetime(2024, 3, 11, tzinfo=timezone.utc),
        )
        assert out == []

    def test_fetch_handles_api_error_status(self, monkeypatch):
        fetcher = NewsFetcher(api_key="dummy")
        monkeypatch.setattr(
            fetcher,
            "_http_get",
            lambda url, timeout=10: json.dumps(
                {"status": "error", "message": "apiKeyInvalid"}
            ),
        )
        out = fetcher.fetch(
            "AAPL",
            start=datetime(2024, 3, 10, tzinfo=timezone.utc),
            end=datetime(2024, 3, 11, tzinfo=timezone.utc),
        )
        assert out == []

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            NewsFetcher(api_key="dummy", source="bloomberg")

    def test_invalid_date_range_raises(self):
        fetcher = NewsFetcher(api_key="dummy")
        with pytest.raises(ValueError):
            fetcher.fetch(
                "AAPL",
                start=datetime(2024, 3, 11, tzinfo=timezone.utc),
                end=datetime(2024, 3, 10, tzinfo=timezone.utc),
            )


# ---------------------------------------------------------------------------
# compute_sentiment_features — end-to-end with mocks
# ---------------------------------------------------------------------------

class _StubFetcher:
    """In-memory news fetcher. Returns pre-canned articles per bar index."""

    def __init__(self, by_index: list[list[dict]]):
        self._by_index = by_index
        self._call = 0

    def fetch(self, symbol, start, end, max_articles=100):  # noqa: ARG002
        i = self._call
        self._call += 1
        if i >= len(self._by_index):
            return []
        return self._by_index[i]


class TestComputeSentimentFeatures:
    def test_columns_and_length(self):
        bar_ts = pd.date_range("2024-03-10", periods=5, freq="D", tz="UTC")
        # Bar 0: positive article; bar 2: negative article; others empty.
        by_index = [
            [
                {
                    "title": "AAPL surge on earnings beat",
                    "description": "",
                    "timestamp": bar_ts[0].to_pydatetime(),
                    "source": "x",
                    "url": "y",
                }
            ],
            [],
            [
                {
                    "title": "Stocks plunge on macro fears",
                    "description": "miss everywhere",
                    "timestamp": bar_ts[2].to_pydatetime(),
                    "source": "x",
                    "url": "y",
                }
            ],
            [],
            [],
        ]
        fetcher = _StubFetcher(by_index)
        model = MockSentimentModel()
        df = compute_sentiment_features(
            "AAPL",
            news_fetcher=fetcher,
            sentiment_model=model,
            bar_timestamps=bar_ts,
            lookback_hours=24.0,
        )
        assert list(df.columns) == [
            "sentiment_score",
            "sentiment_mom_1d",
            "sentiment_mom_3d",
            "article_count",
        ]
        assert len(df) == 5
        # Bar 0 positive, bar 2 negative, empties are 0.
        assert df["sentiment_score"].iloc[0] > 0
        assert df["sentiment_score"].iloc[2] < 0
        assert df["sentiment_score"].iloc[1] == 0.0
        # Article counts.
        assert df["article_count"].iloc[0] == 1
        assert df["article_count"].iloc[1] == 0
        assert df["article_count"].iloc[2] == 1

    def test_no_articles_gives_neutral(self):
        bar_ts = pd.date_range("2024-03-10", periods=3, freq="D", tz="UTC")
        fetcher = _StubFetcher([[], [], []])
        model = MockSentimentModel()
        df = compute_sentiment_features(
            "AAPL",
            news_fetcher=fetcher,
            sentiment_model=model,
            bar_timestamps=bar_ts,
            lookback_hours=24.0,
        )
        assert (df["sentiment_score"] == 0.0).all()
        assert (df["article_count"] == 0).all()

    def test_model_not_called_when_no_articles(self):
        bar_ts = pd.date_range("2024-03-10", periods=3, freq="D", tz="UTC")
        fetcher = _StubFetcher([[], [], []])
        model = MockSentimentModel()
        compute_sentiment_features(
            "AAPL",
            news_fetcher=fetcher,
            sentiment_model=model,
            bar_timestamps=bar_ts,
            lookback_hours=24.0,
        )
        assert model.calls == []

    def test_invalid_lookback(self):
        bar_ts = pd.date_range("2024-03-10", periods=3, freq="D", tz="UTC")
        with pytest.raises(ValueError):
            compute_sentiment_features(
                "AAPL",
                news_fetcher=_StubFetcher([[]]),
                sentiment_model=MockSentimentModel(),
                bar_timestamps=bar_ts,
                lookback_hours=0.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
