"""
NLP Sentiment Features (Jansen — ML for Algorithmic Trading, Ch. 14–16)

Two pieces:
    1. An LLM-based sentiment scorer (FinBERT) with a duck-typed interface
       that lets tests substitute a deterministic mock.
    2. A news-fetch + aggregate pipeline that produces per-bar sentiment
       features: score, multi-window momentum, and article count.

The heavy dependencies (``transformers``, ``torch``) are imported lazily so
that importing this module never triggers a FinBERT download. The ~400MB
FinBERT weights are downloaded on first call to
``FinBERTSentimentModel.predict`` and cached by HuggingFace.

HTTP calls in NewsFetcher go through ``_http_get`` which tests can
monkeypatch — no live network access required.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Structural type: anything with .predict(list[str]) -> list[dict]
# ---------------------------------------------------------------------------

class SentimentScorer(Protocol):
    """Anything that can classify text into sentiment probabilities.

    Structural type used so tests can inject a deterministic mock without
    depending on the real FinBERT pipeline.
    """

    def predict(self, texts: list[str]) -> list[dict]:
        """Return one probability dict per input text."""
        ...


# ---------------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------------

class FinBERTSentimentModel:
    """
    Thin wrapper around HuggingFace's ProsusAI/finbert text classifier.

    The ~400MB model is downloaded on first ``predict`` call and cached by
    the HuggingFace transformers library. The transformers/torch imports
    themselves are lazy so this class can be constructed (and the rest of
    the sentiment module imported) without either package installed.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None  # lazily built on first predict

    def _load(self):
        """Build the HuggingFace pipeline on demand."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FinBERTSentimentModel requires `transformers` and `torch`. "
                "Install them (they are listed in requirements.txt under "
                "Phase 2) or inject a mock sentiment scorer for tests."
            ) from exc
        from transformers import pipeline

        device_idx = 0 if self.device != "cpu" else -1
        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=device_idx,
            top_k=None,  # return all three class scores per input
        )

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of texts into positive/negative/neutral probabilities.

        Args:
            texts: List of strings. Empty list → empty output.

        Returns:
            One dict per input with keys ``positive``, ``negative``,
            ``neutral`` (float probs summing to 1) and ``sentiment_score``
            (= positive - negative, in [-1, 1]).
        """
        if not texts:
            return []
        self._load()
        pipe = self._pipeline
        assert pipe is not None, "_load() must populate _pipeline"

        raw = pipe(list(texts), batch_size=self.batch_size)
        # The pipeline returns either list[list[dict]] (when top_k=None) or
        # list[dict] (single class). Normalize.
        return [self._scores_to_dict(entry) for entry in raw]

    @staticmethod
    def _scores_to_dict(entry) -> dict:
        """Convert a pipeline entry to the standard output dict."""
        if isinstance(entry, dict):
            # Single top-1 result: we only know the max class, pin the rest to 0.
            lbl = entry.get("label", "").lower()
            sc = float(entry.get("score", 0.0))
            probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            if lbl in probs:
                probs[lbl] = sc
            return {**probs, "sentiment_score": probs["positive"] - probs["negative"]}

        probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for d in entry:
            lbl = d["label"].lower()
            if lbl in probs:
                probs[lbl] = float(d["score"])
        score = probs["positive"] - probs["negative"]
        return {**probs, "sentiment_score": score}


# ---------------------------------------------------------------------------
# Exponential-decay aggregation
# ---------------------------------------------------------------------------

def aggregate_sentiment(
    scores: list[dict],
    timestamps: list[datetime],
    half_life_hours: float = 24.0,
) -> float:
    """
    Collapse a list of sentiment scores into one number via exponential decay.

    weight_i = exp(-dt_i / half_life_hours), where dt_i is the gap in hours
    between ``timestamps[i]`` and the most recent timestamp. The aggregate
    is the weight-normalized mean of the ``sentiment_score`` fields.

    Args:
        scores:          List of dicts each containing ``sentiment_score``.
        timestamps:      Parallel list of datetimes.
        half_life_hours: Decay constant (hours). Smaller → more weight on
                         recent items.

    Returns:
        Aggregated sentiment in [-1, 1], or 0.0 if the input is empty.
    """
    if not scores:
        return 0.0
    if len(scores) != len(timestamps):
        raise ValueError("scores and timestamps must have equal length")
    if half_life_hours <= 0:
        raise ValueError("half_life_hours must be positive")

    t_max = max(timestamps)
    weights = np.array(
        [
            math.exp(
                -max(0.0, (t_max - ts).total_seconds()) / 3600.0 / half_life_hours
            )
            for ts in timestamps
        ],
        dtype=float,
    )
    values = np.array([float(s["sentiment_score"]) for s in scores], dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float((weights * values).sum() / total)


# ---------------------------------------------------------------------------
# Sentiment momentum
# ---------------------------------------------------------------------------

def sentiment_momentum(
    sentiment_series: pd.Series,
    windows: Iterable[int] = (1, 3, 7),
) -> pd.DataFrame:
    """
    Rate-of-change features on a sentiment series over multiple lags.

    Output columns: ``sentiment_mom_{w}`` for each w in ``windows``. Each is
    ``series[t] - series[t-w]``.

    Args:
        sentiment_series: Per-bar sentiment scores.
        windows:          Positive integer lags.

    Returns:
        pd.DataFrame indexed like the input.
    """
    if any(w < 1 for w in windows):
        raise ValueError("windows must contain only positive integers")
    data = {
        f"sentiment_mom_{w}": sentiment_series - sentiment_series.shift(w)
        for w in windows
    }
    return pd.DataFrame(data, index=sentiment_series.index)


# ---------------------------------------------------------------------------
# News fetcher
# ---------------------------------------------------------------------------

class NewsFetcher:
    """
    Minimal news fetcher (NewsAPI-only for now).

    HTTP goes through ``_http_get`` so tests can monkeypatch network calls.
    Errors (rate-limited, network, JSON) log a warning and return [].
    """

    NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, source: str = "newsapi") -> None:
        if source != "newsapi":
            raise ValueError(f"unsupported source: {source!r}")
        self.api_key = api_key
        self.source = source

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        max_articles: int = 100,
    ) -> list[dict]:
        """Fetch articles about ``symbol`` in [start, end]."""
        if end < start:
            raise ValueError("end must be >= start")
        if max_articles <= 0:
            return []

        params = {
            "q": symbol,
            "from": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "pageSize": str(min(max_articles, 100)),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key,
        }
        url = f"{self.NEWSAPI_ENDPOINT}?{urllib.parse.urlencode(params)}"

        try:
            raw = self._http_get(url)
            payload = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"NewsFetcher: request failed for {symbol}: {exc}")
            return []

        if payload.get("status") != "ok":
            logger.warning(
                f"NewsFetcher: NewsAPI error for {symbol}: "
                f"{payload.get('message', payload)}"
            )
            return []

        out: list[dict] = []
        for a in payload.get("articles", [])[:max_articles]:
            try:
                ts_str = a.get("publishedAt", "")
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "timestamp": ts,
                    "source": (a.get("source") or {}).get("name", ""),
                    "url": a.get("url") or "",
                }
            )
        return out

    # Extracted so tests can monkeypatch without touching the network.
    def _http_get(self, url: str, timeout: float = 10.0) -> str:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Per-bar sentiment feature pipeline
# ---------------------------------------------------------------------------

def compute_sentiment_features(
    symbol: str,
    news_fetcher: NewsFetcher,
    sentiment_model: SentimentScorer,
    bar_timestamps: pd.DatetimeIndex,
    lookback_hours: float = 48.0,
    half_life_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Build per-bar sentiment features for ``symbol``.

    For each bar t: fetch articles in [t - lookback_hours, t], score them,
    aggregate with exponential decay, and emit ``sentiment_score`` +
    ``article_count``. Momentum columns ``sentiment_mom_1d`` and
    ``sentiment_mom_3d`` are computed from the resulting score series over
    1- and 3-bar lags (intended use: daily bars).

    If no articles are found in the window, sentiment defaults to 0.0
    (neutral) and article_count is 0.

    Args:
        symbol:          Instrument ticker (used as the NewsAPI query).
        news_fetcher:    NewsFetcher (or mock with the same interface).
        sentiment_model: Anything with ``.predict(texts) -> list[dict]``.
        bar_timestamps:  Index of per-bar timestamps to produce features at.
        lookback_hours:  News window size.
        half_life_hours: Decay constant for aggregation.

    Returns:
        pd.DataFrame indexed by ``bar_timestamps`` with columns
        sentiment_score, sentiment_mom_1d, sentiment_mom_3d, article_count.
    """
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")

    scores = np.zeros(len(bar_timestamps), dtype=float)
    counts = np.zeros(len(bar_timestamps), dtype=int)

    for i, ts in enumerate(bar_timestamps):
        # Make the timestamp timezone-aware (assume UTC if naive) to match
        # NewsAPI/NewsFetcher expectations.
        ts_aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        start = ts_aware - timedelta(hours=lookback_hours)

        articles = news_fetcher.fetch(symbol, start=start, end=ts_aware)
        if not articles:
            continue

        texts = [
            f"{a.get('title', '')}. {a.get('description', '')}".strip(". ").strip()
            or " "  # avoid empty strings, which some tokenizers choke on
            for a in articles
        ]
        timestamps = [a["timestamp"] for a in articles]
        preds = sentiment_model.predict(texts)
        scores[i] = aggregate_sentiment(preds, timestamps, half_life_hours)
        counts[i] = len(articles)

    score_series = pd.Series(scores, index=bar_timestamps, name="sentiment_score")
    mom_df: pd.DataFrame = sentiment_momentum(score_series, windows=[1, 3])
    mom_1d = mom_df["sentiment_mom_1"]
    mom_3d = mom_df["sentiment_mom_3"]
    out = pd.DataFrame(
        {
            "sentiment_score": score_series,
            "sentiment_mom_1d": mom_1d,
            "sentiment_mom_3d": mom_3d,
            "article_count": counts,
        },
        index=bar_timestamps,
    )
    return out
