"""Orchestrate one scraping iteration: navigate every URL, extract, persist."""
from __future__ import annotations

import json as _json
import logging
import time as _time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from playwright.sync_api import Error as PlaywrightError, sync_playwright
from sqlalchemy import select

from scraper.config import Settings
from scraper.db import session_scope
from scraper.models import (
    Event, OddsSnapshot, RankingSnapshot, Result, ScrapeRun, utcnow,
)
from scraper.parser import (
    TeamRanking,
    VirtualEvent,
    XHR_URL_PATTERNS,
    parse_from_dom,
    parse_from_embedded_json,
    parse_from_xhr_payload,
    parse_ranking_payload,
)
from scraper.utils import hash_payload

log = logging.getLogger("scraper.collector")


def _match_key(ev: VirtualEvent) -> str:
    """Clé unique par occurrence : compétition + paire + timestamp arrondi minute.
    NE PAS inclure round_info car il peut changer entre l'upcoming (0) et le
    fini (16, 17, ...), créant des duplicates. Le timestamp suffit à dédupliquer
    car chaque match a un expected_start unique pour la paire.
    """
    ts = ev.expected_start.strftime("%Y%m%d%H%M") if ev.expected_start else "0"
    return f"{ev.competition}|{ev.team_a}|{ev.team_b}|{ts}"


def _naive_utc(dt):
    """Normalise un datetime en UTC NAÏF (tzinfo retiré) pour un stockage SQLite
    cohérent. Sans ça, certains events sont tz-aware ('+00:00') et d'autres naïfs,
    ce qui casse pd.to_datetime/les comparaisons côté lecture."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        from datetime import timezone as _tz
        return dt.astimezone(_tz.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# MODE API DIRECTE (multi-ligues) — pas de navigateur, pas de timeouts networkidle.
# L'API Sporty-Tech répond aux GET simples avec Origin/Referer bet261.
# Endpoints par ligue :
#   /api/instantleagues/{id}/matches            → round imminent complet + ids des futurs
#   /api/instantleagues/{id}/matches?roundId=N  → les 10 matchs d'un round futur
#   /api/instantleagues/{id}/results?skip=0&take=4
#   /api/instantleagues/{id}/ranking
# ---------------------------------------------------------------------------
_API_BASE = "https://hg-event-api-prod.sporty-tech.net/api/instantleagues"


def _api_get(url: str, user_agent: str, timeout: int = 30, retries: int = 2):
    """GET JSON avec gzip (payloads ~10x plus petits sur connexion lente)."""
    import gzip as _gzip
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "Origin": "https://bet261.mg",
                "Referer": "https://bet261.mg/",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "") == "gzip":
                    raw = _gzip.decompress(raw)
                return _json.loads(raw.decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                _time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _fetch_league_api(league_id: str, user_agent: str
                       ) -> tuple[list[VirtualEvent], list[TeamRanking]]:
    """Récupère matches (round courant + futurs), results, ranking d'UNE ligue."""
    events: list[VirtualEvent] = []
    rankings: list[TeamRanking] = []
    base = f"{_API_BASE}/{league_id}"

    # 1. /matches → round imminent (matchs+cotes inline) + liste de TOUS les rounds
    # futurs (id + expectedStart + eventCategoryId), mais SANS leurs matchs.
    src_matches = f"{base}/matches"
    payload = _api_get(src_matches, user_agent)
    events.extend(parse_from_xhr_payload(payload, src_matches))

    # 2. Rounds FUTURS via le BON endpoint (celui que l'UI utilise sur clic d'onglet) :
    #    /round/{id}?eventCategoryId={cat}&getNext=false  → 1 round complet (10 matchs,
    #    tous les marchés). ⚠️ NE PAS utiliser /matches?roundId=N : ce param est IGNORÉ
    #    et renvoie toujours le round imminent (→ doublons). On prend TOUS les rounds
    #    futurs listés (pas de cap). eventCategoryId est propre à la saison/ligue courante.
    rounds = payload.get("rounds", []) if isinstance(payload, dict) else []
    # ⚠️ CHAQUE round porte SON PROPRE eventCategoryId. Aux transitions de saison,
    # /matches mélange 2 saisons (cat A pour les rounds courants, cat B pour la
    # suivante). Utiliser un cat GLOBAL fait échouer les fetches du mauvais cat
    # -> on récupérait 1 round au lieu de 10. On prend donc le cat de CHAQUE round.
    future_rounds = [r for r in rounds
                     if not r.get("matches") and r.get("id") is not None
                     and r.get("eventCategoryId") is not None]

    def _fetch_round(r: dict) -> list[VirtualEvent]:
        rid, rcat = r.get("id"), r.get("eventCategoryId")
        src_round = f"{_API_BASE}/round/{rid}?eventCategoryId={rcat}&getNext=false"
        try:
            # 1 retry si l'API renvoie un round VIDE (matches:[]) de façon transitoire
            ro = None
            for attempt in range(2):
                rp = _api_get(src_round, user_agent)
                ro = rp.get("round") if isinstance(rp, dict) else None
                if isinstance(ro, dict) and ro.get("matches"):
                    break
                _time.sleep(0.3)
            if not isinstance(ro, dict) or not ro.get("matches"):
                return []  # round pas encore publié côté serveur
            if not ro.get("expectedStart") or str(ro.get("expectedStart")).startswith("0001"):
                ro["expectedStart"] = r.get("expectedStart")
            evs2 = parse_from_xhr_payload({"rounds": [ro]}, src_round)
            # ⚠️ Le payload /round wrappé perd le contexte de ligue -> le parser tague
            # competition='InstantLeague' (sans suffixe). On force la bonne ligue, sinon
            # tous les rounds futurs sont invisibles pour les lectures filtrées par ligue.
            for x in evs2:
                x.competition = f"InstantLeague-{league_id}"
            return evs2
        except Exception as exc:  # noqa: BLE001
            log.warning("league %s round/%s fetch failed: %s", league_id, rid, exc)
            return []

    # Parallélise les fetches /round (rapidité : ~9 appels en parallèle vs séquentiel).
    if future_rounds:
        with ThreadPoolExecutor(max_workers=min(10, len(future_rounds))) as pool:
            for evs2 in pool.map(_fetch_round, future_rounds):
                events.extend(evs2)

    # 3. résultats récents (take=12 ≈ 25 min d'historique : amortit les coupures réseau)
    src_results = f"{base}/results?skip=0&take=12"
    try:
        events.extend(parse_from_xhr_payload(_api_get(src_results, user_agent), src_results))
    except Exception as exc:  # noqa: BLE001
        log.warning("league %s results fetch failed: %s", league_id, exc)

    # 4. classement
    src_ranking = f"{base}/ranking"
    try:
        rankings.extend(parse_ranking_payload(_api_get(src_ranking, user_agent), src_ranking))
    except Exception as exc:  # noqa: BLE001
        log.warning("league %s ranking fetch failed: %s", league_id, exc)

    return events, rankings


def _fetch_all_api(settings: Settings) -> tuple[list[VirtualEvent], list[TeamRanking]]:
    """Toutes les ligues en parallèle (3 workers) via l'API directe."""
    all_events: list[VirtualEvent] = []
    all_rankings: list[TeamRanking] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_fetch_league_api, lid, settings.user_agent): lid
                   for lid in settings.league_ids}
        for fut in as_completed(futures):
            lid = futures[fut]
            try:
                evs, rks = fut.result()
                all_events.extend(evs)
                all_rankings.extend(rks)
                log.info("league %s: %d events, %d rankings (api)", lid, len(evs), len(rks))
            except Exception as exc:  # noqa: BLE001
                failed.append(lid)
                log.error("league %s: API fetch failed entirely: %s", lid, exc)
    if failed:
        log.warning("leagues failed via API: %s", failed)
    return all_events, all_rankings


def _fetch_all(settings: Settings) -> tuple[list[VirtualEvent], list[TeamRanking]]:
    """API directe d'abord (rapide, multi-ligues) ; fallback Playwright si tout échoue."""
    try:
        events, rankings = _fetch_all_api(settings)
        if events or rankings:
            return events, rankings
        log.warning("API directe n'a rien retourné — fallback Playwright")
    except Exception as exc:  # noqa: BLE001
        log.error("API directe en échec global (%s) — fallback Playwright", exc)
    return _fetch_all_playwright(settings)


def _fetch_all_playwright(settings: Settings) -> tuple[list[VirtualEvent], list[TeamRanking]]:
    """Visit all configured URLs in one browser session, route XHR by fragment."""
    captures: list[tuple[str, str, Any]] = []  # (visit_url, xhr_url, payload)
    current = [""]  # mutable closure cell for the active visit URL

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=settings.headless)
        context = browser.new_context(user_agent=settings.user_agent)
        page = context.new_page()
        page.set_default_timeout(settings.page_timeout_ms)

        def on_response(response):
            if not any(pat in response.url for pat in XHR_URL_PATTERNS):
                return
            try:
                ctype = (response.headers.get("content-type") or "").lower()
                if "json" not in ctype:
                    return
                captures.append((current[0], response.url, response.json()))
                log.info("captured XHR visit=%s xhr=%s", current[0], response.url)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not decode XHR %s: %s", response.url, exc)

        page.on("response", on_response)

        events_html_fallback: list[VirtualEvent] = []

        for url in settings.all_urls:
            current[0] = url
            log.info("navigating url=%s", url)
            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(5000)
            except PlaywrightError as exc:
                log.error("navigation failed url=%s err=%s", url, exc)
                continue

            # Pour la page /matches : cliquer sur chaque onglet de round futur
            # pour capturer les XHRs /api/instantleagues/round/N
            if "/matches" in url:
                try:
                    # Retire les overlays qui bloquent le click
                    page.evaluate(
                        "document.querySelectorAll('hg-privacy, [class*=privacy], "
                        "[class*=cookie]').forEach(e => e.remove())"
                    )
                    page.wait_for_timeout(500)
                    tabs = page.query_selector_all("text=/^\\d{2}:\\d{2}$/")
                    log.info("found %d future round tabs", len(tabs))
                    for i, tab in enumerate(tabs):
                        try:
                            tab.click(force=True, timeout=3000)
                            page.wait_for_timeout(1500)
                        except Exception as ex:
                            log.warning("tab %d click failed: %s", i, ex)
                except Exception as ex:
                    log.warning("failed to click future tabs: %s", ex)

            # DOM fallback only triggered for primary URL if nothing else worked
            if url == settings.target_url and not captures:
                html = page.content()
                events_html_fallback = parse_from_embedded_json(html, url)
                if not events_html_fallback:
                    events_html_fallback = parse_from_dom(page, url)

        context.close()
        browser.close()

    all_events: list[VirtualEvent] = []
    all_rankings: list[TeamRanking] = []

    for visit_url, xhr_url, payload in captures:
        if "/ranking" in xhr_url:
            all_rankings.extend(parse_ranking_payload(payload, visit_url))
        else:
            all_events.extend(parse_from_xhr_payload(payload, visit_url))

    if not all_events and not all_rankings:
        all_events = events_html_fallback

    log.info(
        "extracted events=%d rankings=%d via XHR captures=%d",
        len(all_events), len(all_rankings), len(captures),
    )
    return all_events, all_rankings


def _start_run() -> int:
    with session_scope() as session:
        run = ScrapeRun()
        session.add(run)
        session.flush()
        return run.id


def _merge_event(session, ev: VirtualEvent) -> Event:
    """Upsert an Event by match_key. Promote external_id when discovered."""
    key = _match_key(ev)
    event = session.scalar(select(Event).where(Event.match_key == key))
    if event is None:
        event = Event(
            match_key=key,
            external_id=ev.external_id,
            sport=ev.sport,
            competition=ev.competition,
            team_a=ev.team_a,
            team_b=ev.team_b,
            round_info=ev.round_info,
            expected_start=_naive_utc(ev.expected_start),
            source_url=ev.source_url,
        )
        session.add(event)
        session.flush()
    else:
        # /results feed has id=0; fill external_id once /matches feed brings the real one
        if ev.external_id and not event.external_id:
            event.external_id = ev.external_id
        if ev.sport and not event.sport:
            event.sport = ev.sport
        # 🆕 Promouvoir round_info : si on n'a que '0' / vide et que le nouveau payload a une vraie journée
        if ev.round_info and ev.round_info != "0":
            if not event.round_info or event.round_info == "0":
                event.round_info = ev.round_info
        # Mettre a jour expected_start vers le PLUS RECENT (prochaine occurrence),
        # toujours en UTC NAÏF pour un stockage cohérent.
        new_es = _naive_utc(ev.expected_start)
        if new_es is not None:
            current = _naive_utc(event.expected_start)
            if current is None or new_es > current:
                event.expected_start = new_es
    return event


def _persist(events: list[VirtualEvent], rankings: list[TeamRanking], run_id: int) -> None:
    with session_scope() as session:
        run = session.get(ScrapeRun, run_id)
        if run is None:
            raise RuntimeError(f"ScrapeRun {run_id} disappeared")

        # Dédup du batch par match_key : le round-robin + le chevauchement avec le
        # feed /results créent des match_keys en double dans un même batch -> churn de
        # contrainte. On garde la version la + riche (celle qui porte un score).
        dedup: dict[str, VirtualEvent] = {}
        for ev in events:
            if not ev.team_a or not ev.team_b or ev.round_info is None:
                continue
            # Ignorer tout match sans vraie date (sentinelle '0001-01-01' des rounds
            # futurs pas encore publiés) -> évite un match_key fantôme dupliqué ensuite.
            if ev.expected_start is None or ev.expected_start.year < 2000:
                continue
            k = _match_key(ev)
            prev = dedup.get(k)
            if prev is None or (ev.score_a is not None and prev.score_a is None):
                dedup[k] = ev

        for ev in dedup.values():
            # Savepoint PAR EVENT : un échec isolé (collision de contrainte, doublon de
            # résultat) n'annule que CET event, pas tout le batch. Corrige la perte
            # silencieuse des rounds futurs (90 fetchés -> 0 persistés sans erreur).
            try:
                with session.begin_nested():
                    event = _merge_event(session, ev)
                    run.events_seen += 1

                    if any(v is not None for v in (ev.odds_home, ev.odds_draw, ev.odds_away)):
                        odds_payload = {
                            "odds_home": ev.odds_home,
                            "odds_draw": ev.odds_draw,
                            "odds_away": ev.odds_away,
                            "status": ev.status,
                            "extra": {k: v for k, v in ev.extra_markets.items()
                                      if k not in ("goals", "halfTimeScore")},
                        }
                        content_hash = hash_payload({**odds_payload, "event": event.id})
                        already = session.scalar(
                            select(OddsSnapshot).where(
                                OddsSnapshot.event_id == event.id,
                                OddsSnapshot.content_hash == content_hash,
                            )
                        )
                        if already is None:
                            session.add(OddsSnapshot(
                                event_id=event.id,
                                status=ev.status,
                                odds_home=ev.odds_home,
                                odds_draw=ev.odds_draw,
                                odds_away=ev.odds_away,
                                extra_markets=odds_payload["extra"] or None,
                                content_hash=content_hash,
                                scrape_run_id=run_id,
                            ))
                            run.snapshots_inserted += 1

                    if ev.score_a is not None and ev.score_b is not None:
                        # ne pas rattacher un résultat à un event vieux de >6h (suspect)
                        from datetime import timezone as _tz_check
                        too_old = False
                        if event.expected_start:
                            exp = event.expected_start
                            if exp.tzinfo is None:
                                exp = exp.replace(tzinfo=_tz_check.utc)
                            too_old = (utcnow() - exp).total_seconds() / 3600 > 6
                        if not too_old:
                            existing = session.scalar(
                                select(Result).where(Result.event_id == event.id)
                            )
                            if existing is None:
                                session.add(Result(
                                    event_id=event.id,
                                    score_a=ev.score_a,
                                    score_b=ev.score_b,
                                    raw_score=ev.raw_score,
                                    ht_score_a=ev.ht_score_a,
                                    ht_score_b=ev.ht_score_b,
                                    goals_json=ev.goals,
                                    scrape_run_id=run_id,
                                ))
                                run.results_inserted += 1
                            else:
                                if existing.ht_score_a is None and ev.ht_score_a is not None:
                                    existing.ht_score_a = ev.ht_score_a
                                    existing.ht_score_b = ev.ht_score_b
                                if existing.goals_json is None and ev.goals:
                                    existing.goals_json = ev.goals
            except Exception as exc:  # noqa: BLE001
                log.warning("persist: %s vs %s ignoré: %s", ev.team_a, ev.team_b, exc)

        for r in rankings:
            rank_payload = {
                "position": r.position,
                "points": r.points,
                "won": r.won,
                "lost": r.lost,
                "draw": r.draw,
                "history": r.history,
            }
            content_hash = hash_payload({**rank_payload, "team": r.team_name, "comp": r.competition})
            already = session.scalar(
                select(RankingSnapshot).where(
                    RankingSnapshot.team_name == r.team_name,
                    RankingSnapshot.competition == r.competition,
                    RankingSnapshot.content_hash == content_hash,
                )
            )
            if already is None:
                session.add(RankingSnapshot(
                    competition=r.competition,
                    team_name=r.team_name,
                    position=r.position,
                    points=r.points,
                    won=r.won,
                    lost=r.lost,
                    draw=r.draw,
                    history=r.history,
                    content_hash=content_hash,
                    scrape_run_id=run_id,
                ))
                run.rankings_inserted += 1

        run.status = "ok"
        run.finished_at = utcnow()


def _mark_error(run_id: int, message: str) -> None:
    with session_scope() as session:
        run = session.get(ScrapeRun, run_id)
        if run is not None:
            run.status = "error"
            run.error_message = message[:2000]
            run.finished_at = utcnow()


def run_iteration(settings: Settings) -> None:
    run_id = _start_run()
    try:
        events, rankings = _fetch_all(settings)
        _persist(events, rankings, run_id)
        log.info("iteration ok run_id=%d", run_id)
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        log.exception("iteration failed run_id=%d", run_id)
        _mark_error(run_id, f"{type(exc).__name__}: {exc}")
