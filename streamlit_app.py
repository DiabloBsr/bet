"""APP CLOUD â€” Streamlit Community Cloud, SANS base locale.

Fetch LIVE de l'API Bet261 (rounds Ã  venir des 9 ligues) + moteur MARCHÃ‰,
champion certifiÃ© du tournoi d'algos 2026-07 sur 7/8 marchÃ©s (dÃ©vig calibrÃ© ;
Over 2.5 maison = champion du seul marchÃ© non cotÃ©).
Le suivi forward, les sentinelles et le scraper restent sur la machine locale.
"""
from __future__ import annotations
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import streamlit as st

from scraper.parser import parse_from_xhr_payload
from scraper.predictor_v2 import grid_top_k_scores, market_score_grid
import predict_trio as pt

# calibration embarquÃ©e dans le repo (data/ n'est pas versionnÃ©)
if pt._CALIB is None:
    try:
        pt._CALIB = np.asarray(json.loads(
            (ROOT / "config" / "score_calibration.json").read_text(encoding="utf-8"))["correction"], float)
    except Exception:
        pass

API = "https://hg-event-api-prod.sporty-tech.net/api/instantleagues"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
MADA = timezone(timedelta(hours=3))
LEAGUES = {"ðŸ´ Angleterre": "8035", "ðŸŒ Coupe du Monde": "8065", "ðŸ† Champions": "8056",
           "ðŸŒ CAN": "8060", "ðŸ‡®ðŸ‡¹ Italie": "8036", "ðŸ‡ªðŸ‡¸ Espagne": "8037",
           "ðŸ‡«ðŸ‡· France": "8042", "ðŸ‡©ðŸ‡ª Allemagne": "8043", "ðŸ‡µðŸ‡¹ Portugal": "8044"}


def api_get(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip",
        "Origin": "https://bet261.mg", "Referer": "https://bet261.mg/"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode())


@st.cache_data(ttl=40, show_spinner=False)
def fetch_upcoming(lid: str):
    """Matchs Ã  venir (tous les rounds publiÃ©s) d'une ligue, via l'API publique."""
    evs = parse_from_xhr_payload(api_get(f"{API}/{lid}/matches"), f"{API}/{lid}/matches")
    now = datetime.now(timezone.utc)
    out = []
    for e in evs:
        if not e.odds_home or not e.odds_away or e.expected_start is None or e.score_a is not None:
            continue
        es = e.expected_start
        if es.tzinfo is None:
            es = es.replace(tzinfo=timezone.utc)
        if es < now - timedelta(minutes=2):
            continue
        out.append({"team_a": e.team_a, "team_b": e.team_b, "es": es,
                    "oh": float(e.odds_home), "od": float(e.odds_draw), "oa": float(e.odds_away),
                    "xm": e.extra_markets or {}, "lid": lid})
    return sorted(out, key=lambda m: m["es"])


def to_match(ev) -> dict:
    """Construit l'objet match (board + probas champion) pour l'affichage/combos."""
    oh, od, oa = ev["oh"], ev["od"], ev["oa"]
    inv = 1/oh + 1/od + 1/oa
    x12 = ((1/oh)/inv, (1/od)/inv, (1/oa)/inv)
    cons, top1c = [], None
    try:
        g = market_score_grid(ev["xm"].get("Score exact"))
        if g is not None:
            cons = [(s, float(p)) for s, p in grid_top_k_scores(g, 8)]
            cal = pt._apply_calib(dict(cons))
            top1c = max(cal.items(), key=lambda kv: kv[1]) if cal else None
    except Exception:
        pass
    local = ev["es"].astimezone(MADA).strftime("%H:%M")
    return {"match": f"{ev['team_a']} v {ev['team_b']}", "local": local,
            "cotes": (oh, od, oa), "x12": x12,
            "over25_pct": pt._over25_calib(oh, od, oa),
            "consensus_top3": cons[:3], "top1_calibre": top1c,
            "board": pt.market_board(ev["xm"], oh, od, oa)}


def round_label(es: datetime) -> str:
    """Heure locale Mada HH:MM d'un round."""
    return es.astimezone(MADA).strftime("%H:%M")


def render_round(lg_name: str, want) -> None:
    """PrÃ©dit un round : le prochain si `want` est None, sinon celui de l'heure Mada saisie."""
    with st.spinner("Fetch API Bet261â€¦"):
        try:
            ups = fetch_upcoming(LEAGUES[lg_name])
        except Exception as exc:
            st.error(f"API injoignable depuis le cloud : {exc}")
            return
    if not ups:
        st.info("Aucun round Ã  venir publiÃ© pour cette ligue (rÃ©essaie dans ~1 min).")
        return

    # ---- sÃ©lection du round cible ----
    rounds = sorted({e["es"] for e in ups})
    if want is None:
        target_es = rounds[0]
    else:
        tgt = want.strftime("%H:%M")
        target_es = next((es for es in rounds if round_label(es) == tgt), None)
        if target_es is None:
            st.warning(f"Aucun round publiÃ© Ã  **{tgt}** (heure Mada) sur cette ligue. "
                       "Rounds disponibles : " + " Â· ".join(round_label(es) for es in rounds))
            return

    matches = [to_match(e) for e in ups if e["es"] == target_es]
    left = int((target_es - datetime.now(timezone.utc)).total_seconds())
    cd = f"â³ coup d'envoi dans {max(left,0)//60}:{max(left,0)%60:02d}" if left > 0 else "ðŸ”´ imminent"
    st.success(f"Round {matches[0]['local']} Mada â€” {len(matches)} matchs   {cd}")
    if len(rounds) > 1:
        st.caption("Autres rounds publiÃ©s : "
                   + " Â· ".join(round_label(es) for es in rounds if es != target_es))

    for m in matches:
        ph, pd_, pa = m["x12"]
        k1, k2, k3 = st.columns([3, 2, 3])
        with k1:
            st.markdown(f"**{m['match']}**  \n`{m['cotes'][0]}/{m['cotes'][1]}/{m['cotes'][2]}`")
            st.markdown(f"1 **{ph*100:.0f}%** Â· X {pd_*100:.0f}% Â· 2 **{pa*100:.0f}%**")
        with k2:
            if m["top1_calibre"]:
                st.metric("Top-1 (calibrÃ©)", m["top1_calibre"][0],
                          f"{m['top1_calibre'][1]*100:.0f}%")
            st.caption("Top-3 : " + " Â· ".join(s for s, _ in m["consensus_top3"])
                       if m["consensus_top3"] else "")
            if m["over25_pct"] is not None:
                st.caption(f"âš½ Over 2.5 : **{m['over25_pct']}%**")
        with k3:
            board = m["board"]
            best = sorted(((mk, s, p, o) for mk, rows in board.items()
                           for s, p, o in rows if p >= 0.55), key=lambda r: -r[2])[:3]
            if best:
                st.caption("ðŸŽ¯ " + " Â· ".join(f"**{s[:20]}** [{mk}] {p*100:.0f}% ({o:g})"
                                              for mk, s, p, o in best))
        with st.expander(f"ðŸ“‹ Tous les marchÃ©s â€” {m['match']} (âœ… = â‰¥55%)"):
            for mkt, rows in board.items():
                top = rows[:6] if len(rows) > 8 else rows
                st.markdown(f"**{mkt}** : " + " Â· ".join(
                    f"{'âœ… ' if p >= 0.55 else ''}{s} **{p*100:.0f}%** ({o:g})"
                    for s, p, o in top))
        st.divider()

    # ---- combinÃ© conseillÃ© ----
    st.subheader("ðŸŽ¯ CombinÃ© conseillÃ© â€” max-gain / min-risque")
    f1, f2, f3, f4 = st.columns(4)
    fam = f1.selectbox("Famille", ["SÃ»rs", "âš½ TOTALS"], index=1)
    tgt_odds = f2.number_input("Cote cible min", 1.5, 20.0, 3.0, 0.5)
    mx = int(f3.selectbox("Jambes max", ["2", "3"], index=1))
    inter = f4.toggle("ðŸŒ 9 ligues", value=False)
    pool = matches
    if inter:
        with st.spinner("Fetch des 9 liguesâ€¦"):
            with ThreadPoolExecutor(max_workers=9) as ex:
                allups = sum(ex.map(lambda l: (fetch_upcoming(l) or [])[:10],
                                    LEAGUES.values()), [])
        # fenÃªtre standard de 6 min, Ã©tendue jusqu'au round visÃ© s'il est plus lointain
        horizon = max(datetime.now(timezone.utc) + timedelta(minutes=6),
                      target_es + timedelta(minutes=1))
        pool = [to_match(e) for e in allups if e["es"] <= horizon]
        st.caption(f"pool inter-ligues : {len(pool)} matchs (jusqu'Ã  {round_label(horizon)} Mada)")
    if fam.startswith("âš½"):
        combos = pt.build_combos(pool, float(tgt_odds), mx, markets=pt.TOTALS_MARKETS,
                                 min_legs=1, p_min=0.20)
    else:
        combos = pt.build_combos(pool, float(tgt_odds), mx)
    if combos:
        for i, c in enumerate(combos, 1):
            st.markdown(f"**Option {i} â€” cote {c['odds']:.2f} Â· rÃ©ussite {c['p']*100:.0f}%** "
                        f"Â· espÃ©rance {c['ev']*100:+.1f}%")
            for (mn, mkt, s, p, o) in c["legs"]:
                st.caption(f"  â€¢ {mn} â€” {mkt} : **{s}** ({p*100:.0f}%, cote {o:g})")
    else:
        st.caption("Aucun combinÃ© n'atteint la cote cible sur ce round.")
    st.caption("âš ï¸ Politique : Ã  cote cible fixÃ©e, le combinÃ© le PLUS PROBABLE. Aucun pari n'a "
               "d'espÃ©rance positive (marge du book ~6%/jambe simple ; RNG calibrÃ© â€” prouvÃ© sur "
               "17 campagnes / ~2 000 tests). Outil d'aide, pas une promesse de gain.")


def main():
    st.set_page_config(page_title="Bet261 Virtual â€” LIVE", page_icon="âš½", layout="wide")
    st.title("âš½ Bet261 Virtual Football â€” prÃ©diction LIVE (cloud)")
    st.caption("Moteur MARCHÃ‰ certifiÃ© champion (tournoi d'algos, 32k matchs OOS) : probas dÃ©vigÃ©es "
               "calibrÃ©es + Over 2.5 maison. DonnÃ©es en direct de l'API â€” aucun serveur local requis.")
    now_mada = datetime.now(MADA)
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.metric("ðŸ• Heure Mada (UTC+3)", now_mada.strftime("%d/%m %H:%M"))
    lg_name = c2.selectbox("Ligue", list(LEAGUES), index=0)
    auto = c3.toggle("ðŸ”„ Suivi auto (~45s)", value=False)

    # ---- choix du round Ã  prÃ©dire : prochain, ou heure prÃ©cise saisie ----
    r1, r2 = st.columns([2, 2])
    mode = r1.radio("Round Ã  prÃ©dire", ["â­ï¸ Prochain round", "ðŸ• Heure prÃ©cise (Mada)"],
                    horizontal=True)
    want = None
    if mode.startswith("ðŸ•"):
        default_t = (now_mada + timedelta(minutes=5)).replace(second=0, microsecond=0).time()
        want = r2.time_input("Heure du round (HH:MM, heure Mada)", value=default_t, step=60)
        try:  # aperÃ§u des rounds actuellement publiÃ©s, pour saisir une heure valide
            prev = fetch_upcoming(LEAGUES[lg_name])
            if prev:
                r2.caption("Rounds publiÃ©s : " + " Â· ".join(
                    round_label(es) for es in sorted({e["es"] for e in prev})))
        except Exception:
            pass

    if st.button("ðŸ”® PrÃ©dire le round", type="primary") or auto:
        render_round(lg_name, want)

    if auto:
        time.sleep(45)
        st.rerun()


main()