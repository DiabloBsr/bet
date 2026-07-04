"""APP CLOUD — Streamlit Community Cloud, SANS base locale.

Fetch LIVE de l'API Bet261 (round imminent des 9 ligues) + moteur MARCHÉ,
champion certifié du tournoi d'algos 2026-07 sur 7/8 marchés (dévig calibré ;
Over 2.5 maison = champion du seul marché non coté).
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

# calibration embarquée dans le repo (data/ n'est pas versionné)
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
LEAGUES = {"🏴 Angleterre": "8035", "🌍 Coupe du Monde": "8065", "🏆 Champions": "8056",
           "🌍 CAN": "8060", "🇮🇹 Italie": "8036", "🇪🇸 Espagne": "8037",
           "🇫🇷 France": "8042", "🇩🇪 Allemagne": "8043", "🇵🇹 Portugal": "8044"}


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
    """Matchs à venir (round imminent) d'une ligue, via l'API publique."""
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


def main():
    st.set_page_config(page_title="Bet261 Virtual — LIVE", page_icon="⚽", layout="wide")
    st.title("⚽ Bet261 Virtual Football — prédiction LIVE (cloud)")
    st.caption("Moteur MARCHÉ certifié champion (tournoi d'algos, 32k matchs OOS) : probas dévigées "
               "calibrées + Over 2.5 maison. Données en direct de l'API — aucun serveur local requis.")
    now_mada = datetime.now(MADA)
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.metric("🕐 Heure Mada (UTC+3)", now_mada.strftime("%d/%m %H:%M"))
    lg_name = c2.selectbox("Ligue", list(LEAGUES), index=0)
    auto = c3.toggle("🔄 Suivi auto (~45s)", value=False)

    if st.button("🔮 Prédire le prochain round", type="primary") or auto:
        with st.spinner("Fetch API Bet261…"):
            try:
                ups = fetch_upcoming(LEAGUES[lg_name])
            except Exception as exc:
                st.error(f"API injoignable depuis le cloud : {exc}")
                st.stop()
        if not ups:
            st.info("Aucun round à venir publié pour cette ligue (réessaie dans ~1 min).")
            st.stop()
        first_es = ups[0]["es"]
        matches = [to_match(e) for e in ups if e["es"] == first_es]
        left = int((first_es - datetime.now(timezone.utc)).total_seconds())
        cd = f"⏳ coup d'envoi dans {max(left,0)//60}:{max(left,0)%60:02d}" if left > 0 else "🔴 imminent"
        st.success(f"Round {matches[0]['local']} Mada — {len(matches)} matchs   {cd}")

        for m in matches:
            ph, pd_, pa = m["x12"]
            k1, k2, k3 = st.columns([3, 2, 3])
            with k1:
                st.markdown(f"**{m['match']}**  \n`{m['cotes'][0]}/{m['cotes'][1]}/{m['cotes'][2]}`")
                st.markdown(f"1 **{ph*100:.0f}%** · X {pd_*100:.0f}% · 2 **{pa*100:.0f}%**")
            with k2:
                if m["top1_calibre"]:
                    st.metric("Top-1 (calibré)", m["top1_calibre"][0],
                              f"{m['top1_calibre'][1]*100:.0f}%")
                st.caption("Top-3 : " + " · ".join(s for s, _ in m["consensus_top3"])
                           if m["consensus_top3"] else "")
                if m["over25_pct"] is not None:
                    st.caption(f"⚽ Over 2.5 : **{m['over25_pct']}%**")
            with k3:
                board = m["board"]
                best = sorted(((mk, s, p, o) for mk, rows in board.items()
                               for s, p, o in rows if p >= 0.55), key=lambda r: -r[2])[:3]
                if best:
                    st.caption("🎯 " + " · ".join(f"**{s[:20]}** [{mk}] {p*100:.0f}% ({o:g})"
                                                  for mk, s, p, o in best))
            with st.expander(f"📋 Tous les marchés — {m['match']} (✅ = ≥55%)"):
                for mkt, rows in board.items():
                    top = rows[:6] if len(rows) > 8 else rows
                    st.markdown(f"**{mkt}** : " + " · ".join(
                        f"{'✅ ' if p >= 0.55 else ''}{s} **{p*100:.0f}%** ({o:g})"
                        for s, p, o in top))
            st.divider()

        # ---- combiné conseillé ----
        st.subheader("🎯 Combiné conseillé — max-gain / min-risque")
        f1, f2, f3, f4 = st.columns(4)
        fam = f1.selectbox("Famille", ["Sûrs", "⚽ TOTALS"], index=1)
        tgt = f2.number_input("Cote cible min", 1.5, 20.0, 3.0, 0.5)
        mx = int(f3.selectbox("Jambes max", ["2", "3"], index=1))
        inter = f4.toggle("🌍 9 ligues", value=False)
        pool = matches
        if inter:
            with st.spinner("Fetch des 9 ligues…"):
                with ThreadPoolExecutor(max_workers=9) as ex:
                    allups = sum(ex.map(lambda l: (fetch_upcoming(l) or [])[:10],
                                        LEAGUES.values()), [])
            horizon = datetime.now(timezone.utc) + timedelta(minutes=6)
            pool = [to_match(e) for e in allups if e["es"] <= horizon]
            st.caption(f"pool inter-ligues : {len(pool)} matchs (~6 prochaines minutes)")
        if fam.startswith("⚽"):
            combos = pt.build_combos(pool, float(tgt), mx, markets=pt.TOTALS_MARKETS,
                                     min_legs=1, p_min=0.20)
        else:
            combos = pt.build_combos(pool, float(tgt), mx)
        if combos:
            for i, c in enumerate(combos, 1):
                st.markdown(f"**Option {i} — cote {c['odds']:.2f} · réussite {c['p']*100:.0f}%** "
                            f"· espérance {c['ev']*100:+.1f}%")
                for (mn, mkt, s, p, o) in c["legs"]:
                    st.caption(f"  • {mn} — {mkt} : **{s}** ({p*100:.0f}%, cote {o:g})")
        else:
            st.caption("Aucun combiné n'atteint la cote cible sur ce round.")
        st.caption("⚠️ Politique : à cote cible fixée, le combiné le PLUS PROBABLE. Aucun pari n'a "
                   "d'espérance positive (marge du book ~6%/jambe simple ; RNG calibré — prouvé sur "
                   "17 campagnes / ~2 000 tests). Outil d'aide, pas une promesse de gain.")

    if auto:
        time.sleep(45)
        st.rerun()


main()
