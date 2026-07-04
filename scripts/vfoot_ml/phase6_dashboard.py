"""VFoot-ML — PHASE 6 : Dashboard de suivi (Streamlit).

Rassemble tout le système :
  • État du MONITEUR DE DÉRIVE (scellé / fenêtre détectée) — le signal qui compte.
  • PRÉDICTIONS LIVE des rounds à venir (1X2, score, Over/Under, value, Kelly, confiance).
  • TENDANCE de calibration dans le temps (l'écart réalisé-implicite par ligue).
  • PERFORMANCE des modèles (Phase 3) et du backtest (Phase 4).

Lancement :
    streamlit run scripts/vfoot_ml/phase6_dashboard.py
Déployable gratuitement sur Streamlit Community Cloud.

Les fonctions load_*/get_* sont PURES (testables sans serveur) ; seul le bloc UI
en bas appelle streamlit.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))       # pour importer predict_v2v5

DATA = ROOT / "data"


# ====================================================================== #
# FONCTIONS DE DONNÉES (pures, sans streamlit -> testables)
# ====================================================================== #
def load_monitor() -> dict | None:
    p = DATA / "edge_monitor_report.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def load_history() -> pd.DataFrame:
    """Aplati edge_monitor_history.jsonl : 1 ligne par (run, ligue)."""
    p = DATA / "edge_monitor_history.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        for lg, v in (r.get("leagues") or {}).items():
            rows.append({"run": r.get("run_utc"), "data_clock": r.get("data_clock"),
                         "league": lg, "gap_recent_pp": v.get("gap_recent_pp"),
                         "over35_roi_recent": v.get("over35_roi_recent"), "n_arb": v.get("n_arb")})
    return pd.DataFrame(rows)


def load_model_results() -> dict:
    out = {}
    for key, fn in [("phase3", "phase3_results.json"), ("phase4", "phase4_results.json")]:
        p = DATA / "vfoot_ml" / fn
        try:
            out[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except Exception:
            out[key] = None
    return out


def get_live_predictions(limit: int = 10):
    """Retourne (liste de rapports, df). Robuste : [] si scraper/modèles absents."""
    try:
        from phase5_live import LivePredictor
        return LivePredictor().predict_upcoming(limit=limit)
    except Exception as exc:
        return {"error": str(exc)}, None


def get_round_predictions(ts_utc_prefix: str):
    """Prédit le round à une heure UTC donnée ('YYYY-MM-DD HH:MM')."""
    try:
        from phase5_live import LivePredictor
        return LivePredictor().predict_round(ts_utc_prefix)
    except Exception as exc:
        return {"error": str(exc)}, None


def v2v5_fit_uncached():
    """Fit V5+V2 (lourd) — à envelopper dans st.cache_resource. Retourne (eng,m5,v2,n)."""
    from scraper.config import load_settings
    from sqlalchemy import create_engine
    import predict_v2v5 as pvv
    eng = create_engine(load_settings().db_url)
    m5, v2, n = pvv.fit(eng)
    return eng, m5, v2, n


def get_v2v5_round(models, target=None):
    """Croisement V2×V5 sur un round (heure Mada HH:MM ou None=prochain)."""
    import predict_v2v5 as pvv
    eng, m5, v2, _n = models
    return pvv.predict_round(eng, m5, v2, target)


def monitor_summary(mon: dict) -> tuple[bool, list[dict]]:
    """(any_flag, [{league, sealed, gap_recent, gap_full, n_arb, flags}])."""
    if not mon:
        return False, []
    rows = []
    for lg, v in (mon.get("leagues") or {}).items():
        rows.append({"league": lg, "sealed": v.get("sealed", True),
                     "gap_recent_pp": v.get("gap_over35_recent_pp"),
                     "gap_full_pp": v.get("gap_over35_full_pp"),
                     "n_arb": v.get("n_arb", 0), "flags": v.get("flags", [])})
    return bool(mon.get("any_flag")), rows


# ====================================================================== #
# UI STREAMLIT
# ====================================================================== #
def render():
    import streamlit as st

    from datetime import datetime, timedelta, timezone

    st.set_page_config(page_title="VFoot-ML Dashboard", page_icon="⚽", layout="wide")
    st.title("⚽ VFoot-ML — Système de prédiction & veille")

    now_local = datetime.now(timezone.utc) + timedelta(hours=3)   # heure MADAGASCAR (UTC+3)
    mon0 = load_monitor()
    data_clock = "?"
    h = load_history()
    if not h.empty and h["data_clock"].notna().any():
        try:
            dc = pd.to_datetime(h["data_clock"].dropna().iloc[-1]) + pd.Timedelta(hours=3)
            data_clock = dc.strftime("%d/%m %H:%M")
        except Exception:
            pass
    cA, cB = st.columns(2)
    cA.metric("🕐 Heure Mada (UTC+3)", now_local.strftime("%d/%m/%Y %H:%M"))
    cB.metric("📥 Données fraîches jusqu'à", data_clock)
    st.caption("RNG Bet261 — prédiction calibrée + détection de fenêtre d'attaque")

    tab_live, tab_cotes, tab_mon, tab_models = st.tabs(
        ["🔴 Live", "💰 Gros cotes", "📡 Moniteur de dérive", "📊 Modèles & Backtest"])

    # -------- GROS COTES PROBABLES --------
    with tab_cotes:
        st.subheader("💰 Gros cotes probables (cote ≥ 3)")
        st.caption("Outcomes à grosse cote classés par PROBABILITÉ du modèle calibré. "
                   "🟢 EV>0 (rare), 🔴 EV<0. Honnête : sur un book calibré, presque tout est 🔴 — "
                   "ici on montre les MOINS pires / les plus probables, pas des pièges.")
        if st.button("🔮 Chercher les gros cotes probables"):
            preds, _ = get_live_predictions(20)
            if isinstance(preds, dict) and preds.get("error"):
                st.error(preds["error"])
            elif not preds:
                st.info("Aucun round à venir capté (le scraper doit tourner).")
            else:
                rows = []
                for r in preds:
                    if r.get("error"):
                        continue
                    for b in r.get("details", {}).get("value_bets", []):
                        rows.append({"heure": r.get("heure_locale"), "match": r["match"],
                                     "marché": b["marche"], "pari": b["pari"], "cote": b["cote"],
                                     "proba %": b["proba"], "EV %": b["ev"]})
                if not rows:
                    st.info("Aucune cote ≥ 3 exploitable sur ces rounds.")
                else:
                    dfb = pd.DataFrame(rows).sort_values("proba %", ascending=False).reset_index(drop=True)
                    st.dataframe(dfb.head(40), use_container_width=True, hide_index=True)
                    pos = dfb[dfb["EV %"] > 0]
                    if len(pos):
                        st.success(f"🟢 {len(pos)} paris à EV>0 (à CONFIRMER — probablement du bruit sur un book calibré).")
                        st.dataframe(pos, use_container_width=True, hide_index=True)
                    else:
                        st.warning("🔴 Aucun pari à EV>0 — le book est calibré, tout perd à long terme. "
                                   "Le tableau montre les plus PROBABLES, pas des +EV.")

    # -------- MONITEUR (le plus important) --------
    with tab_mon:
        mon = load_monitor()
        any_flag, rows = monitor_summary(mon)
        if not mon:
            st.warning("Aucun rapport de moniteur. Lance `python scripts/edge_monitor.py`.")
        elif any_flag:
            st.error("⚠️ **FENÊTRE POTENTIELLE DÉTECTÉE** — un marché a flagué. "
                     "Vérifier en OOS (workflow adverse) AVANT toute mise.")
        else:
            st.success("🔒 **Toutes les ligues SCELLÉES** — aucun edge, capital protégé. "
                       "Le système veille et alertera si ça change.")
        if rows:
            dfm = pd.DataFrame(rows)
            st.dataframe(dfm, use_container_width=True, hide_index=True)
        hist = load_history()
        if not hist.empty:
            st.subheader("Tendance : écart réalisé−implicite (Over 3.5) par ligue")
            piv = hist.pivot_table(index="run", columns="league", values="gap_recent_pp")
            st.line_chart(piv)
            st.caption("Un écart qui s'éloigne durablement de 0 (>3pp) = fenêtre qui s'ouvre.")

    # -------- LIVE (CROISEMENT V2 × V5) --------
    with tab_live:
        st.subheader("🔀 Prédiction — CROISEMENT V2 × V5")
        st.caption("V5 (Poisson FT + HT/FT + marché) × V2 (grille blendée) → consensus tranché par les deux moteurs.")
        try:
            models = st.cache_resource(v2v5_fit_uncached)()
            st.caption(f"✓ V5+V2 fittés sur {models[3]} matchs (mis en cache).")
        except Exception as exc:
            st.error(f"Fit V2×V5 impossible : {exc}")
            models = None

        def _render_croise(res):
            if not res or not res.get("matches"):
                st.info("Aucun match à venir capté (le scraper doit tourner).")
                return
            st.success(f"Round {res.get('target')} Mada — {len(res['matches'])} matchs")
            for m in res["matches"]:
                ph, pd_, pa = m["x12"]
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                with c1:
                    st.markdown(f"**{m['match']}**  \n`{m['cotes'][0]}/{m['cotes'][1]}/{m['cotes'][2]}`")
                    st.markdown(f"1 **{ph*100:.0f}%** · X {pd_*100:.0f}% · 2 **{pa*100:.0f}%**")
                    st.caption("🤝 V2 & V5 d'accord" if m.get("accord") else "🔀 V2 ≠ V5 (prudence)")
                with c2:
                    cs = m.get("consensus_top3") or [("?", 0)]
                    st.metric("Consensus", cs[0][0], f"{cs[0][1]*100:.0f}%")
                    if len(cs) > 1:
                        st.caption("ou " + " · ".join(s for s, _ in cs[1:3]))
                    ov = m.get("over25_pct")
                    if ov is not None:
                        st.caption(f"⚽ Over 2.5 : **{ov}%** (calibré)")
                with c3:
                    st.caption("**V5** :  \n" + "  \n".join(f"{s} ({p*100:.0f}%)" for s, p in m.get("v5_top3", [])))
                with c4:
                    st.caption("**V2** :  \n" + "  \n".join(f"{s} ({p*100:.0f}%)" for s, p in m.get("v2_top3", [])))
                st.divider()

        import re as _re2
        cT, cB1 = st.columns([3, 1])
        t_str = cT.text_input("Heure Mada du round (ex: 21:03) — laisse vide pour le prochain",
                              value="", key="rtime")
        go_h = cB1.button("🎯 Ce round")
        go_now = st.button("🔮 Prédire le prochain round à venir")
        if (go_h or go_now) and models:
            target = None
            if go_h and t_str.strip():
                _d = _re2.findall(r"\d+", t_str)
                if len(_d) >= 2:
                    target = f"{int(_d[0]) % 24:02d}:{int(_d[1]) % 60:02d}"
            with st.spinner("Croisement V2×V5 en cours…"):
                res = get_v2v5_round(models, target)
            if target and res.get("rounds") and target not in res["rounds"]:
                st.warning(f"Round {target} non dispo. Rounds captés : {res['rounds'][:10]}")
            _render_croise(res)
        st.info("Prédicteur Live = croisement V2×V5 (les deux moteurs). VFoot-ML gardé intact, non utilisé ici.")

    # -------- MODÈLES --------
    with tab_models:
        mr = load_model_results()
        if mr.get("phase3"):
            st.subheader("Phase 3 — comparaison OOS des modèles")
            md = pd.DataFrame(mr["phase3"]["models"]).T
            st.dataframe(md, use_container_width=True)
            st.caption("Benchmark = probas des cotes. Aucun modèle ne le bat -> cote imbattable.")
        if mr.get("phase4"):
            st.subheader("Phase 4 — backtest des stratégies")
            sd = pd.DataFrame(mr["phase4"]["strategies"])
            st.dataframe(sd, use_container_width=True, hide_index=True)
            mc = mr["phase4"].get("monte_carlo_best", {})
            if mc:
                st.metric("Probabilité de ruine (Monte-Carlo)", f"{mc.get('p_ruine_pct')}%")


if __name__ == "__main__":
    # Si lancé via `streamlit run`, le module streamlit est présent -> on rend l'UI.
    try:
        import streamlit  # noqa
        render()
    except ModuleNotFoundError:
        # exécution simple `python phase6_dashboard.py` -> auto-test des fonctions de données
        print("Streamlit absent — auto-test des fonctions de données :")
        mon = load_monitor(); af, rows = monitor_summary(mon)
        print(f"  moniteur: {'OK' if mon else 'absent'} | any_flag={af} | {len(rows)} ligues")
        print(f"  historique: {len(load_history())} lignes")
        mr = load_model_results()
        print(f"  modèles: phase3={'OK' if mr.get('phase3') else '-'} phase4={'OK' if mr.get('phase4') else '-'}")
        print("  -> lance le dashboard avec : streamlit run scripts/vfoot_ml/phase6_dashboard.py")
