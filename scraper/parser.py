"""Extract virtual sports events from the target page.

Trois stratégies sont tentées en cascade, de la plus stable à la plus fragile :

  1. parse_from_xhr_payload  — JSON capturé sur les réponses Fetch/XHR
  2. parse_from_embedded_json — JSON injecté dans le HTML initial (__NEXT_DATA__, etc.)
  3. parse_from_dom           — sélecteurs CSS sur le DOM rendu (fallback)

Configurer le parser pour un nouveau site :
  - Ouvrir DevTools → onglet Network → filtre Fetch/XHR → recharger.
    Noter un fragment d'URL stable de la réponse contenant les cotes et
    l'ajouter à XHR_URL_PATTERNS.
  - Si pas d'XHR : View-source, chercher des balises <script> contenant
    un JSON volumineux ; ajouter leur id à EMBEDDED_JSON_KEYS.
  - En dernier recours : ajuster DOM_SELECTORS via l'inspecteur d'éléments,
    en privilégiant les attributs data-* stables (data-testid, data-event-id).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constantes de configuration — éditer ici quand la cible change.
# ---------------------------------------------------------------------------

XHR_URL_PATTERNS: list[str] = [
    "/api/instantleagues/",  # bet261.mg / Sporty-Tech instant league matches feed
]

EMBEDDED_JSON_KEYS: list[str] = [
    "__NEXT_DATA__",
    "__INITIAL_STATE__",
    "__APP_DATA__",
]

DOM_SELECTORS: dict[str, str] = {
    "event_card": "[data-testid='event-card']",
    "sport_name": "[data-role='sport']",
    "competition": "[data-role='competition']",
    "team_a": "[data-role='team-home']",
    "team_b": "[data-role='team-away']",
    "odds_home": "[data-market='1x2'] [data-outcome='1']",
    "odds_draw": "[data-market='1x2'] [data-outcome='X']",
    "odds_away": "[data-market='1x2'] [data-outcome='2']",
    "score": "[data-role='score']",
    "status": "[data-role='status']",
    "round_info": "[data-role='round']",
    "external_id_attr_node": "[data-event-id]",
    "external_id_attr_name": "data-event-id",
}


# ---------------------------------------------------------------------------
# Structure de sortie commune aux trois stratégies.
# ---------------------------------------------------------------------------

@dataclass
class VirtualEvent:
    source_url: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    external_id: str | None = None
    sport: str | None = None
    competition: str | None = None
    team_a: str | None = None
    team_b: str | None = None
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    extra_markets: dict[str, Any] = field(default_factory=dict)
    score_a: int | None = None
    score_b: int | None = None
    raw_score: str | None = None
    # Mi-temps : score a la 45e min
    ht_score_a: int | None = None
    ht_score_b: int | None = None
    # Liste des buts avec timing
    goals: list | None = None
    status: str | None = None
    round_info: str | None = None
    expected_start: datetime | None = None


# ---------------------------------------------------------------------------
# Helpers internes.
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


def _parse_score(raw: str | None) -> tuple[int | None, int | None]:
    if not raw:
        return None, None
    match = re.search(r"(\d+)\s*[-:]\s*(\d+)", raw)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _walk_to_event_list(payload: Any) -> list[dict[str, Any]]:
    """BFS dans le payload pour trouver la première liste qui ressemble à des events."""
    queue: list[Any] = [payload]
    marker_keys = {"id", "eventId", "homeTeam", "teamA", "competitors", "competition"}
    while queue:
        node = queue.pop(0)
        if isinstance(node, list) and node and isinstance(node[0], dict):
            if set(node[0].keys()) & marker_keys:
                return node
            queue.extend(node)
        elif isinstance(node, dict):
            queue.extend(node.values())
    return []


def _get_team(item: dict[str, Any], side: str) -> str | None:
    keys = {
        "home": ["homeTeam", "teamA", "home", "competitorA"],
        "away": ["awayTeam", "teamB", "away", "competitorB"],
    }[side]
    for k in keys:
        value = item.get(k)
        if isinstance(value, dict):
            name = value.get("name")
            if name:
                return name
        elif isinstance(value, str) and value:
            return value
    competitors = item.get("competitors")
    if isinstance(competitors, list) and len(competitors) >= 2:
        idx = 0 if side == "home" else 1
        c = competitors[idx]
        if isinstance(c, dict):
            return c.get("name")
    return None


def _get_odd(item: dict[str, Any], outcome: str) -> Any:
    markets = item.get("markets") or item.get("odds") or {}
    if isinstance(markets, dict):
        m1x2 = markets.get("1X2") or markets.get("matchOdds") or markets
        if isinstance(m1x2, dict):
            return m1x2.get(outcome) or m1x2.get(f"outcome_{outcome}")
    if isinstance(markets, list):
        for m in markets:
            if isinstance(m, dict) and m.get("outcome") == outcome:
                return m.get("price") or m.get("odd")
    return None


# ---------------------------------------------------------------------------
# Stratégie 1 — JSON XHR.
# ---------------------------------------------------------------------------

def parse_from_xhr_payload(payload: Any, source_url: str) -> list[VirtualEvent]:
    """Mappe un payload JSON vers des VirtualEvent.

    Detecte 3 schemas Sporty-Tech :
      - /matches : `{rounds: [...], betTypes: [...]}` (multi-rounds)
      - /results : `{rounds: [...], hasMore: ...}`
      - /round/N : `{round: {matches: [...]}}` (single round futur)
    """
    # /round/N : wrapper {round: {...}} - on transforme en {rounds: [{...}]}
    if (
        isinstance(payload, dict)
        and "round" in payload
        and isinstance(payload["round"], dict)
        and "matches" in payload["round"]
    ):
        payload = {"rounds": [payload["round"]]}

    # Sporty-Tech : feeds /matches (rounds + betTypes) et /results (rounds + hasMore)
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("rounds"), list)
        and payload["rounds"]
        and isinstance(payload["rounds"][0], dict)
        and "matches" in payload["rounds"][0]
    ):
        return _parse_sporty_tech(payload, source_url)

    events: list[VirtualEvent] = []
    for item in _walk_to_event_list(payload):
        raw_score = item.get("score") or item.get("result")
        score_a, score_b = _parse_score(raw_score)
        events.append(VirtualEvent(
            source_url=source_url,
            external_id=str(item.get("id") or item.get("eventId") or "") or None,
            sport=item.get("sport") or item.get("sportName"),
            competition=item.get("competition") or item.get("league"),
            team_a=_get_team(item, "home"),
            team_b=_get_team(item, "away"),
            odds_home=_safe_float(_get_odd(item, "1")),
            odds_draw=_safe_float(_get_odd(item, "X")),
            odds_away=_safe_float(_get_odd(item, "2")),
            extra_markets={k: v for k, v in item.items() if k.startswith("market_")},
            raw_score=raw_score,
            score_a=score_a,
            score_b=score_b,
            status=item.get("status") or item.get("state"),
            round_info=str(item.get("round") or item.get("cycle") or "") or None,
        ))
    return events


def _entry_point_from_url(url: str) -> str | None:
    """Extract entry-point ID from a bet261 page URL or API URL."""
    match = re.search(r"/instant[-_]?leagues?/(\d+)", url)
    return match.group(1) if match else None


def _parse_iso_utc(ts: Any) -> datetime | None:
    """Parse Sporty-Tech ISO timestamps; '0001-01-01T00:00:00Z' = sentinel."""
    if not isinstance(ts, str) or ts.startswith("0001-"):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_sporty_tech(payload: dict, source_url: str) -> list[VirtualEvent]:
    """Mappe les feeds `/api/instantleagues/{id}/matches` ET `/results` (Sporty-Tech).

    Distingue à la volée :
      - matches : présence de `eventBetTypes` → on remplit les cotes
      - results : présence de `score` ou `goals` → on remplit le score final
    Une même équipe/round peut apparaître dans les deux feeds — la dédup se fait
    côté collector via match_key.
    """
    events: list[VirtualEvent] = []
    entry_point = _entry_point_from_url(source_url)
    competition = f"InstantLeague-{entry_point}" if entry_point else "InstantLeague"

    for round_obj in payload.get("rounds", []):
        round_number = round_obj.get("roundNumber")
        round_start = _parse_iso_utc(round_obj.get("expectedStart"))

        for match in round_obj.get("matches", []):
            home = match.get("homeTeam") or {}
            away = match.get("awayTeam") or {}
            team_a = home.get("name")
            team_b = away.get("name")
            if not team_a or not team_b:
                continue
            # Fallback : si roundNumber=0/None, utiliser le champ `round` du match (string "19")
            effective_round = round_number
            if not effective_round or effective_round == 0:
                m_round = match.get("round")
                if m_round:
                    try:
                        effective_round = int(m_round)
                    except (ValueError, TypeError):
                        pass
            if not effective_round:
                continue  # vraiment pas de journée → on ignore

            odds_home = odds_draw = odds_away = None
            extra_markets: dict[str, Any] = {}

            for bt in match.get("eventBetTypes", []):
                bt_name = bt.get("name")
                items = {
                    item.get("shortName"): item.get("odds")
                    for item in bt.get("eventBetTypeItems", [])
                    if item.get("active") and item.get("odds") is not None
                }
                if bt_name == "1X2":
                    odds_home = items.get("1")
                    odds_draw = items.get("X")
                    odds_away = items.get("2")
                elif items:
                    extra_markets[bt_name] = items

            raw_score = match.get("score")
            score_a, score_b = _parse_score(raw_score)
            has_result = score_a is not None and score_b is not None

            # HT score - parser "1:1" -> (1, 1)
            ht_raw = match.get("halfTimeScore")
            ht_a, ht_b = _parse_score(ht_raw)
            if match.get("halfTimeScore"):
                extra_markets["halfTimeScore"] = match.get("halfTimeScore")
            # Goals timing list - kept full structure
            goals_list = match.get("goals") or None
            if goals_list:
                extra_markets["goals"] = goals_list

            match_start = round_start or _parse_iso_utc(match.get("expectedStart"))

            events.append(VirtualEvent(
                source_url=source_url,
                external_id=str(match.get("id")) if match.get("id") else None,
                sport="Virtual Football",
                competition=competition,
                team_a=team_a,
                team_b=team_b,
                odds_home=_safe_float(odds_home),
                odds_draw=_safe_float(odds_draw),
                odds_away=_safe_float(odds_away),
                extra_markets=extra_markets,
                raw_score=raw_score,
                score_a=score_a,
                score_b=score_b,
                ht_score_a=ht_a,
                ht_score_b=ht_b,
                goals=goals_list,
                status="finished" if has_result else "upcoming",
                round_info=str(effective_round),
                expected_start=match_start,
            ))
    return events


# ---------------------------------------------------------------------------
# Ranking (classement de la ligue).
# ---------------------------------------------------------------------------

@dataclass
class TeamRanking:
    competition: str
    captured_at: datetime
    team_name: str
    position: int | None = None
    points: int | None = None
    won: int | None = None
    lost: int | None = None
    draw: int | None = None
    history: list | None = None


def parse_ranking_payload(payload: Any, source_url: str) -> list[TeamRanking]:
    """Map a `{teams: [...]}` ranking payload to TeamRanking instances."""
    if not isinstance(payload, dict) or not isinstance(payload.get("teams"), list):
        return []
    entry_point = _entry_point_from_url(source_url)
    competition = f"InstantLeague-{entry_point}" if entry_point else "InstantLeague"
    now = datetime.now(timezone.utc)
    out: list[TeamRanking] = []
    for t in payload["teams"]:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append(TeamRanking(
            competition=competition,
            captured_at=now,
            team_name=t["name"],
            position=t.get("position"),
            points=t.get("points"),
            won=t.get("won"),
            lost=t.get("lost"),
            draw=t.get("draw"),
            history=t.get("history"),
        ))
    return out


# ---------------------------------------------------------------------------
# Stratégie 2 — JSON embarqué dans le HTML.
# ---------------------------------------------------------------------------

def parse_from_embedded_json(html: str, source_url: str) -> list[VirtualEvent]:
    for key in EMBEDDED_JSON_KEYS:
        pattern = rf'id=["\']{re.escape(key)}["\'][^>]*>(\{{.*?\}})\s*</script>'
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        events = parse_from_xhr_payload(payload, source_url)
        if events:
            return events
    return []


# ---------------------------------------------------------------------------
# Stratégie 3 — DOM rendu.
# ---------------------------------------------------------------------------

def parse_from_dom(page, source_url: str) -> list[VirtualEvent]:
    """Reçoit un Playwright Page et lit les sélecteurs définis dans DOM_SELECTORS."""
    events: list[VirtualEvent] = []
    cards = page.query_selector_all(DOM_SELECTORS["event_card"])

    for card in cards:
        def text(selector: str) -> str | None:
            node = card.query_selector(selector)
            return node.inner_text().strip() if node else None

        ext_id = None
        ext_node = card.query_selector(DOM_SELECTORS["external_id_attr_node"])
        if ext_node:
            ext_id = ext_node.get_attribute(DOM_SELECTORS["external_id_attr_name"])

        raw_score = text(DOM_SELECTORS["score"])
        score_a, score_b = _parse_score(raw_score)

        events.append(VirtualEvent(
            source_url=source_url,
            external_id=ext_id,
            sport=text(DOM_SELECTORS["sport_name"]),
            competition=text(DOM_SELECTORS["competition"]),
            team_a=text(DOM_SELECTORS["team_a"]),
            team_b=text(DOM_SELECTORS["team_b"]),
            odds_home=_safe_float(text(DOM_SELECTORS["odds_home"])),
            odds_draw=_safe_float(text(DOM_SELECTORS["odds_draw"])),
            odds_away=_safe_float(text(DOM_SELECTORS["odds_away"])),
            raw_score=raw_score,
            score_a=score_a,
            score_b=score_b,
            status=text(DOM_SELECTORS["status"]),
            round_info=text(DOM_SELECTORS["round_info"]),
        ))
    return events
