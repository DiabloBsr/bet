"""AUDIT DE FIABILITÉ — « quand on dit X%, ça arrive combien de fois ? »

Diagramme de calibration sur l'historique : on bucket les probabilités affichées
(dévig marché) et on compare à la fréquence RÉELLE. Parfait = la diagonale.
Prouve que les % de l'app sont dignes de confiance (sans améliorer l'accuracy).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LG = "InstantLeague-8035"


def _gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


def compute(engine, sample=25000):
    """Retourne un DataFrame {marché, bucket, proba_moy, freq_reelle, n}."""
    from sqlalchemy import text
    df = pd.read_sql(text(f"""
        SELECT o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
               r.score_a sa, r.score_b sb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND e.competition='{LG}'
          AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
        ORDER BY e.expected_start DESC LIMIT {int(sample)}"""), engine)
    inv = 1/df.oh + 1/df.od + 1/df.oa
    recs = []
    # 1X2 (domicile) : proba implicite vs victoire domicile
    recs += _pairs("1X2 (domicile gagne)", (1/df.oh)/inv, (df.sa > df.sb).astype(int))
    # Over 2.5 via Total de buts
    io25, o25 = [], (df.sa + df.sb > 2.5).astype(int).values
    for raw in df.xm:
        try: xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception: xm = {}
        tt = _gm(xm, "Total de buts")
        v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
             if tt and isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99} if tt else {}
        s = sum(v.values())
        io25.append(sum(v[str(k)] for k in range(3, 7))/s if s and len(v) == 7 else np.nan)
    io25 = np.array(io25); m = ~np.isnan(io25)
    recs += _pairs("Over 2.5 buts", pd.Series(io25[m]), pd.Series(o25[m]))
    return pd.DataFrame(recs)


def _pairs(name, prob, outcome):
    prob = np.asarray(prob, float); outcome = np.asarray(outcome, float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    out = []
    for b in range(10):
        sel = idx == b
        if sel.sum() >= 50:
            out.append({"marché": name, "bucket": f"{10*b}-{10*b+10}%",
                        "proba_moy": round(100*float(prob[sel].mean()), 1),
                        "freq_reelle": round(100*float(outcome[sel].mean()), 1),
                        "n": int(sel.sum())})
    return out


def render(st, engine):
    st.subheader("📏 Fiabilité des probabilités — sont-elles dignes de confiance ?")
    try:
        d = compute(engine)
    except Exception as exc:
        st.caption(f"Calcul indisponible : {exc}"); return
    if not len(d):
        st.caption("Pas assez de données."); return
    for mk in d["marché"].unique():
        sub = d[d["marché"] == mk]
        err = float((sub.proba_moy - sub.freq_reelle).abs().mean())
        st.markdown(f"**{mk}** — écart moyen annoncé vs réel : **{err:.1f}pp** "
                    f"({'✅ très fiable' if err < 2 else '⚠️ écart'})")
        chart = sub.set_index("proba_moy")[["freq_reelle"]].rename(columns={"freq_reelle": "réel %"})
        chart["parfait (=annoncé)"] = sub.proba_moy.values
        st.line_chart(chart)
    st.caption("La courbe 'réel %' doit coller à 'parfait' (la diagonale). Si oui, quand l'app "
               "affiche 60%, ça arrive vraiment ~60% du temps — les chiffres ne mentent pas.")
