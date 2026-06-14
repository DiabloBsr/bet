"""Unit tests for the parser layer (no I/O, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scraper.parser import (
    _parse_score,
    _safe_float,
    _walk_to_event_list,
    parse_from_embedded_json,
    parse_from_xhr_payload,
)


def test_parse_score_handles_separators_and_none():
    assert _parse_score("2-1") == (2, 1)
    assert _parse_score("3 : 0") == (3, 0)
    assert _parse_score(None) == (None, None)
    assert _parse_score("upcoming") == (None, None)


def test_safe_float_accepts_comma_decimals():
    assert _safe_float("1,75") == 1.75
    assert _safe_float("1.85") == 1.85
    assert _safe_float("abc") is None
    assert _safe_float(None) is None


def test_walk_finds_nested_event_list():
    payload = {"data": {"sports": {"items": [
        {"eventId": "x", "homeTeam": "A", "awayTeam": "B"},
    ]}}}
    items = _walk_to_event_list(payload)
    assert len(items) == 1
    assert items[0]["eventId"] == "x"


def test_parse_from_xhr_minimal_event():
    payload = {"events": [{
        "id": "evt-1",
        "sport": "Virtual Football",
        "competition": "Virtual Premier League",
        "homeTeam": {"name": "Reds"},
        "awayTeam": {"name": "Blues"},
        "markets": {"1X2": {"1": 1.85, "X": 3.40, "2": 4.10}},
        "status": "upcoming",
        "round": "R12",
    }]}
    events = parse_from_xhr_payload(payload, "https://example.com")
    assert len(events) == 1
    e = events[0]
    assert e.external_id == "evt-1"
    assert e.team_a == "Reds"
    assert e.team_b == "Blues"
    assert e.odds_home == 1.85
    assert e.odds_draw == 3.40
    assert e.odds_away == 4.10
    assert e.round_info == "R12"
    assert e.status == "upcoming"


def test_parse_from_xhr_with_finished_score():
    payload = [{
        "eventId": "evt-2",
        "homeTeam": "Alpha",
        "awayTeam": "Beta",
        "score": "2-1",
        "status": "finished",
    }]
    events = parse_from_xhr_payload(payload, "https://example.com")
    assert events[0].score_a == 2
    assert events[0].score_b == 1


def test_parse_from_embedded_json_next_data():
    html = """
    <html><body>
    <script id="__NEXT_DATA__" type="application/json">{"props":{"events":[
        {"id":"abc","sport":"Virtual Tennis","homeTeam":"X","awayTeam":"Y",
         "markets":{"1X2":{"1":1.5,"2":2.5}},"status":"live"}
    ]}}</script>
    </body></html>
    """
    events = parse_from_embedded_json(html, "https://example.com")
    assert len(events) == 1
    assert events[0].sport == "Virtual Tennis"
    assert events[0].odds_home == 1.5
    assert events[0].odds_draw is None  # marché non présent
    assert events[0].odds_away == 2.5
