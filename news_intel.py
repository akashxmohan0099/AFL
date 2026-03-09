"""
AFL Prediction Pipeline — News Intelligence Extraction
========================================================
Processes scraped news articles through Claude API to extract structured
intelligence signals: injuries, suspensions, form trends, tactical changes,
expert predictions, and more.

Usage:
    python news_intel.py process              # Process today's articles
    python news_intel.py process --days 3     # Process last N days
    python news_intel.py refresh              # Rebuild latest.json rolling feed
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import config

NEWS_DIR = config.DATA_DIR / "news"
ARTICLES_DIR = NEWS_DIR / "articles"
INTEL_DIR = NEWS_DIR / "intel"

INTEL_ROLLING_DAYS = 7
INTEL_BATCH_SIZE = 5
INTEL_MODEL = "claude-sonnet-4-20250514"
MAX_ARTICLES_PER_RUN = 30

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# 18 canonical team names for the LLM prompt
CANONICAL_TEAMS = [
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood",
    "Essendon", "Fremantle", "Geelong", "Gold Coast",
    "Greater Western Sydney", "Hawthorn", "Melbourne", "North Melbourne",
    "Port Adelaide", "Richmond", "St Kilda", "Sydney",
    "West Coast", "Western Bulldogs",
]

SYSTEM_PROMPT = """You are an AFL analytics intelligence analyst. Your job is to extract structured signals from AFL news articles that are relevant for predicting match outcomes, player performance, and team composition.

For each article, extract a JSON object with these fields:
- "headline": string — the article headline
- "summary": string — 1-2 sentence summary of the key intelligence
- "signal_type": one of ["injury", "suspension", "form", "tactical", "selection", "prediction", "general"]
- "teams": array of canonical team names affected (use ONLY these names: """ + json.dumps(CANONICAL_TEAMS) + """)
- "players": array of player names in "Last, First" format
- "sentiment": one of ["positive", "negative", "neutral", "mixed"]
- "direction": object mapping team or player names to "bullish" or "bearish" or "neutral"
  (e.g. {"Carlton": "bearish", "Cripps, Patrick": "bullish"})
- "key_facts": array of extracted facts (e.g. "Suspended 3 weeks", "Hamstring injury, test", "Named in extended squad")
- "relevance_score": float 0.0-1.0 — how impactful for match/player predictions
  (key player injury/suspension = 0.8-1.0, tactical insight = 0.5-0.7, general preview = 0.2-0.4, historical/off-field = 0.0-0.1)
- "prediction_impact": string — brief description of how this affects predictions

Return a JSON array of objects, one per article. If an article has no prediction-relevant intelligence, set relevance_score to 0.0 and signal_type to "general"."""


def _article_id(url: str) -> str:
    """Generate stable ID from article URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _load_processed_ids() -> set:
    """Load IDs of already-processed articles from intel cache."""
    processed = set()
    if not INTEL_DIR.exists():
        return processed
    for f in INTEL_DIR.glob("intel_*.json"):
        try:
            with open(f) as fp:
                signals = json.load(fp)
            for s in signals:
                if "id" in s:
                    processed.add(s["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return processed


def _load_articles(days: int = 1) -> list[dict]:
    """Load scraped articles from the last N days."""
    articles = []
    today = datetime.now()
    for d in range(days):
        date_str = (today - timedelta(days=d)).strftime("%Y-%m-%d")
        path = ARTICLES_DIR / f"articles_{date_str}.json"
        if path.exists():
            with open(path) as f:
                articles.extend(json.load(f))
    return articles


def process_articles(days: int = 1):
    """Process unprocessed articles through Claude API.

    Reads from article cache, skips already-processed articles,
    batches remaining articles, calls Claude API, saves intel signals.
    """
    INTEL_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Set it to enable intelligence extraction.")
        log.info("Falling back to basic article metadata without LLM analysis.")
        _fallback_process(days)
        return

    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        _fallback_process(days)
        return

    articles = _load_articles(days)
    if not articles:
        log.info("No articles to process")
        return

    processed_ids = _load_processed_ids()
    unprocessed = [
        a for a in articles
        if _article_id(a["url"]) not in processed_ids
    ][:MAX_ARTICLES_PER_RUN]

    if not unprocessed:
        log.info("All articles already processed")
        return

    log.info(f"Processing {len(unprocessed)} unprocessed articles ({len(articles)} total, {len(processed_ids)} already done)")

    client = anthropic.Anthropic(api_key=api_key)
    all_signals = []

    # Process in batches
    for batch_start in range(0, len(unprocessed), INTEL_BATCH_SIZE):
        batch = unprocessed[batch_start:batch_start + INTEL_BATCH_SIZE]
        batch_num = batch_start // INTEL_BATCH_SIZE + 1
        total_batches = (len(unprocessed) + INTEL_BATCH_SIZE - 1) // INTEL_BATCH_SIZE

        # Build prompt
        articles_text = []
        for i, article in enumerate(batch):
            text = article.get("full_text") or article.get("summary") or article["headline"]
            # Truncate long articles
            if len(text) > 2000:
                text = text[:2000] + "..."
            articles_text.append(f"--- Article {i+1} ---\nHeadline: {article['headline']}\nURL: {article['url']}\nContent: {text}")

        user_msg = "Extract intelligence signals from these AFL articles. Return a JSON array.\n\n" + "\n\n".join(articles_text)

        log.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} articles...")

        try:
            response = client.messages.create(
                model=INTEL_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            # Parse response
            response_text = response.content[0].text
            # Find JSON array in response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                signals = json.loads(response_text[json_start:json_end])
            else:
                log.warning(f"  No JSON array found in response for batch {batch_num}")
                continue

            # Enrich signals with metadata
            now = datetime.now().isoformat()
            for i, signal in enumerate(signals):
                if i < len(batch):
                    signal["id"] = _article_id(batch[i]["url"])
                    signal["source_url"] = batch[i]["url"]
                    signal["published_at"] = batch[i].get("published_at") or now
                    signal["processed_at"] = now
                all_signals.append(signal)

        except Exception as e:
            log.error(f"  Batch {batch_num} failed: {e}")
            continue

    if not all_signals:
        log.warning("No signals extracted")
        return

    # Save daily intel file
    date_str = datetime.now().strftime("%Y-%m-%d")
    intel_path = INTEL_DIR / f"intel_{date_str}.json"

    existing = []
    if intel_path.exists():
        with open(intel_path) as f:
            existing = json.load(f)

    existing_ids = {s.get("id") for s in existing}
    new_signals = [s for s in all_signals if s.get("id") not in existing_ids]
    combined = existing + new_signals

    with open(intel_path, "w") as f:
        json.dump(combined, f, indent=2)

    log.info(f"Saved {len(combined)} signals ({len(new_signals)} new) to {intel_path}")

    # Rebuild latest.json
    rebuild_latest()


def _fallback_process(days: int = 1):
    """Basic processing without LLM — extracts signals from article metadata only."""
    articles = _load_articles(days)
    if not articles:
        log.info("No articles to process")
        return

    processed_ids = _load_processed_ids()
    now = datetime.now().isoformat()
    signals = []

    for article in articles:
        aid = _article_id(article["url"])
        if aid in processed_ids:
            continue

        headline = article.get("headline", "")
        headline_lower = headline.lower()

        # Basic signal type detection from headline keywords
        if any(w in headline_lower for w in ["injur", "hamstring", "knee", "calf", "concuss", "shoulder"]):
            signal_type = "injury"
            sentiment = "negative"
        elif any(w in headline_lower for w in ["suspend", "ban", "match review", "tribunal"]):
            signal_type = "suspension"
            sentiment = "negative"
        elif any(w in headline_lower for w in ["return", "fit", "back", "available"]):
            signal_type = "injury"
            sentiment = "positive"
        elif any(w in headline_lower for w in ["form", "star", "best", "domina", "peak"]):
            signal_type = "form"
            sentiment = "positive"
        elif any(w in headline_lower for w in ["team", "select", "named", "lineup", "squad"]):
            signal_type = "selection"
            sentiment = "neutral"
        elif any(w in headline_lower for w in ["tip", "predict", "preview", "expect"]):
            signal_type = "prediction"
            sentiment = "neutral"
        elif any(w in headline_lower for w in ["tactic", "gameplan", "strategy", "role"]):
            signal_type = "tactical"
            sentiment = "neutral"
        else:
            signal_type = "general"
            sentiment = "neutral"

        relevance = 0.6 if signal_type in ("injury", "suspension") else 0.3

        signals.append({
            "id": aid,
            "source_url": article["url"],
            "headline": headline,
            "summary": article.get("summary", ""),
            "signal_type": signal_type,
            "teams": article.get("teams", []),
            "players": article.get("players", []),
            "sentiment": sentiment,
            "direction": {},
            "key_facts": [],
            "relevance_score": relevance,
            "prediction_impact": "",
            "published_at": article.get("published_at") or now,
            "processed_at": now,
        })

    if not signals:
        log.info("No new signals to process")
        return

    # Save
    date_str = datetime.now().strftime("%Y-%m-%d")
    intel_path = INTEL_DIR / f"intel_{date_str}.json"

    existing = []
    if intel_path.exists():
        with open(intel_path) as f:
            existing = json.load(f)

    existing_ids = {s.get("id") for s in existing}
    new_signals = [s for s in signals if s.get("id") not in existing_ids]
    combined = existing + new_signals

    with open(intel_path, "w") as f:
        json.dump(combined, f, indent=2)

    log.info(f"Saved {len(combined)} signals ({len(new_signals)} new, keyword-based) to {intel_path}")
    rebuild_latest()


def rebuild_latest():
    """Rebuild latest.json from recent daily intel + injury data."""
    INTEL_DIR.mkdir(parents=True, exist_ok=True)

    all_signals = []
    cutoff = datetime.now() - timedelta(days=INTEL_ROLLING_DAYS)

    # Load daily intel files
    for f in sorted(INTEL_DIR.glob("intel_*.json")):
        date_str = f.stem.replace("intel_", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                continue
        except ValueError:
            continue
        with open(f) as fp:
            all_signals.extend(json.load(fp))

    # Also inject injury data as signals
    from news import INJURIES_DIR
    injury_files = sorted(INJURIES_DIR.glob("injuries_*.json"))
    if injury_files:
        latest_inj = injury_files[-1]
        inj_date = latest_inj.stem.replace("injuries_", "")
        with open(latest_inj) as f:
            injuries = json.load(f)

        for inj in injuries:
            severity = inj.get("severity", 2)
            signals_sentiment = "negative" if severity >= 2 else "neutral"
            sev_label = ["Test", "1 week", "2-3 weeks", "4-8 weeks", "Season"][min(severity, 4)]
            all_signals.append({
                "id": f"inj_{hashlib.md5((inj.get('team','') + inj.get('player','')).encode()).hexdigest()[:8]}",
                "source_url": "https://www.afl.com.au/matches/injury-list",
                "headline": f"{inj.get('player', 'Unknown')} — {inj.get('injury', 'Unknown')}",
                "summary": f"{inj.get('player')} ({inj.get('team')}) is listed with {inj.get('injury', 'an injury')}. Estimated return: {inj.get('estimated_return', 'TBC')}.",
                "signal_type": "injury",
                "teams": [inj.get("team", "")],
                "players": [inj.get("player", "")],
                "sentiment": signals_sentiment,
                "direction": {inj.get("team", ""): "bearish"} if severity >= 2 else {},
                "key_facts": [f"{inj.get('injury', 'Injury')}: {sev_label}"],
                "relevance_score": min(0.3 + severity * 0.15, 1.0),
                "prediction_impact": f"{'Key' if severity >= 3 else 'Potential'} availability concern for {inj.get('team', 'team')}",
                "published_at": f"{inj_date}T00:00:00",
                "processed_at": datetime.now().isoformat(),
            })

    # Deduplicate by ID (keep latest)
    seen = {}
    for s in all_signals:
        sid = s.get("id", "")
        if sid:
            seen[sid] = s
    all_signals = list(seen.values())

    # Sort by relevance (high first), then by date (newest first)
    all_signals.sort(key=lambda s: (-s.get("relevance_score", 0), s.get("published_at", "")), reverse=False)
    all_signals.sort(key=lambda s: s.get("published_at", ""), reverse=True)

    # Compute summary stats
    by_type = {}
    by_team = {}
    for s in all_signals:
        st = s.get("signal_type", "general")
        by_type[st] = by_type.get(st, 0) + 1
        for t in s.get("teams", []):
            by_team[t] = by_team.get(t, 0) + 1

    latest = {
        "signals": all_signals,
        "total": len(all_signals),
        "by_type": by_type,
        "by_team": dict(sorted(by_team.items())),
        "updated": datetime.now().isoformat(),
    }

    out_path = INTEL_DIR / "latest.json"
    with open(out_path, "w") as f:
        json.dump(latest, f, indent=2)

    log.info(f"Built latest.json: {len(all_signals)} signals ({len(by_type)} types, {len(by_team)} teams)")
    return latest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python news_intel.py [process|refresh]")
        print("  process              Process today's articles through LLM")
        print("  process --days N     Process last N days of articles")
        print("  refresh              Rebuild latest.json from cached intel")
        sys.exit(1)

    cmd = sys.argv[1]
    days = 1
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])

    if cmd == "process":
        process_articles(days=days)
    elif cmd == "refresh":
        rebuild_latest()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
