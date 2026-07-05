"""Dashboard AVIATOR — historique live + audit d'équité + simulateur de stratégies.

streamlit run scripts/aviator/dashboard.py --server.port 8514
"""
from __future__ import annotations
import sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import streamlit as st
from scripts.aviator import audit as A
from scripts.aviator import strategy_sim as S

DB = ROOT / "data" / "aviator.db"


def load():
    if not DB.exists():
        return np.array([])
    db = sqlite3.connect(DB)
    m = np.array([r[0] for r in db.execute(
        "SELECT multiplier FROM aviator_rounds ORDER BY rowid")], float)
    db.close()
    return m


def main():
    st.set_page_config(page_title="Aviator — Audit & Stratégies", page_icon="✈️", layout="wide")
    try:
        from scripts.ui_theme import inject_theme, hero
    except Exception:
        sys.path.insert(0, str(ROOT / "scripts")); from ui_theme import inject_theme, hero
    inject_theme(st, accent="#ef4444", accent2="#f59e0b", accent3="#ec4899")
    hero(st, "✈️ Aviator — Audit & Stratégies",
         "Provably-fair = IMPRÉVISIBLE. Zéro prédiction : audit d'équité réel + vrai risque de chaque cash-out",
         badges=["🔬 audit équité", "🎯 simulateur cash-out", "⚖️ marge mesurée", "🛡️ risque de ruine"])

    m = load()
    if len(m) < 10:
        st.warning(f"Seulement {len(m)} manches collectées. Lance le collecteur "
                   "(`python scripts/aviator/collector_service.py`) et laisse-le tourner — "
                   "l'audit devient fiable vers ~5 000 manches.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manches collectées", f"{len(m)}")
    c2.metric("Médiane", f"{np.median(m):.2f}×")
    c3.metric("Crash instantané", f"{100*(m<1.005).mean():.1f}%")
    e = A.house_edge(m)
    c4.metric("Marge maison estimée", f"{100*e:.1f}%", "Spribe annonce ~3%")

    st.subheader("📊 Derniers résultats")
    st.write(" · ".join(f"**{x:.2f}×**" if x >= 10 else f"{x:.2f}×" for x in m[-30:][::-1]))

    tab_p, tab_a, tab_s, tab_c = st.tabs(
        ["🔮 Prochain round", "🔬 Audit d'équité", "🎯 Simulateur de stratégie", "⚖️ Comparateur"])

    with tab_p:
        st.markdown("### 🔮 Prédiction du prochain round")
        st.error("⚠️ **Le multiplicateur exact est IMPRÉVISIBLE** — le crash est scellé "
                 "cryptographiquement (provably-fair) AVANT ta mise. Personne ne peut le prédire, "
                 "et tout « prédicteur Aviator » qui promet un chiffre est une **arnaque**. "
                 "Voici la SEULE prédiction honnête : la **probabilité** que ton objectif soit atteint.")
        cp = st.columns([2, 2, 3])
        T = cp[0].number_input("Ton objectif de cash-out ×", 1.05, 100.0, 2.0, 0.1)
        _t = cp[1].text_input("Heure du round (optionnel)", value="", placeholder="ex 21:30")
        # proba de survie + IC Wilson
        k = int((m >= T).sum()); n = len(m); p = k / n
        z = 1.96
        cen = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        lo, hi = max(0, cen-half), min(1, cen+half)
        cp[2].metric(f"P(prochain ≥ {T:g}×)", f"{100*p:.1f}%",
                     f"IC95 [{100*lo:.0f}–{100*hi:.0f}%] · n={n}")
        if _t.strip():
            st.caption(f"⏱️ Heure saisie « {_t} » — **ça ne change RIEN** : les manches sont "
                       "i.i.d. (indépendantes). La proba ci-dessus vaut pour N'IMPORTE quel round.")
        # « au plus proche du résultat » : la distribution honnête, par percentiles
        st.markdown("**Le plus proche possible du résultat = la distribution (pas un chiffre)** :")
        qs = [50, 25, 10, 5, 1]
        cols = st.columns(len(qs))
        for c, q in zip(cols, qs):
            thr = np.percentile(m, 100 - q)
            c.metric(f"{q}% de chance ≥", f"{thr:.2f}×")
        st.caption(f"Médiane {np.median(m):.2f}× · un cash-out prudent vise ≤ médiane. "
                   "Ces seuils sont l'estimation la plus proche possible — le reste est du hasard pur.")
        # preuve live que le passé ne prédit pas le futur
        if n > 15:
            lm = np.log(np.clip(m, 1, None))
            a, b = lm[:-1] - lm[:-1].mean(), lm[1:] - lm[1:].mean()
            corr = float((a @ b) / (np.sqrt((a@a)*(b@b)) or 1))
            st.info(f"🔬 Corrélation entre manche N et N+1 : **{corr:+.3f}** "
                    f"({'≈ 0 → le passé ne prédit RIEN' if abs(corr) < 0.15 else 'à surveiller'}). "
                    "C'est la preuve mathématique de l'imprévisibilité — utilise le simulateur pour "
                    "gérer ton RISQUE, la seule variable que tu contrôles.")

    with tab_a:
        st.markdown("**Distribution des multiplicateurs de crash**")
        bins = [1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 1000]
        labels = ["1-1.2", "1.2-1.5", "1.5-2", "2-3", "3-5", "5-10", "10-20", "20-50", "50+"]
        cats = pd.cut(m, bins=bins, labels=labels, right=False)
        st.bar_chart(pd.Series(cats).value_counts().reindex(labels).fillna(0))
        st.markdown("**Survie P(M ≥ x) — réel vs jeu équitable (marge 3%)**")
        rows = [{"seuil": f"{x}×", "réel %": round(100*(m >= x).mean(), 1),
                 "fair %": round(100*0.97/x, 1)} for x in (1.5, 2, 3, 5, 10, 20)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        ac, z = A.independence(m)
        st.caption(f"Indépendance des manches (provably-fair ⇒ i.i.d.) : autocorrélation "
                   f"{', '.join(f'lag{k}={v:+.3f}' for k,v in ac.items())} · runs test z={z:+.2f} "
                   f"({'✅ i.i.d.' if abs(z)<2 else '⚠ à surveiller'})")
        st.info("RTP(T)=T·P(M≥T) doit être ~plat et proche de (1−marge). "
                f"Marge mesurée : **{100*e:.1f}%**. Aucune poche exploitable — le jeu est équitable "
                "au sens statistique, ce qui NE le rend pas gagnable (marge = perte à long terme).")

    with tab_s:
        st.markdown("**Teste une stratégie de cash-out** (Monte-Carlo sur la distribution réelle)")
        cc = st.columns(5)
        strat = cc[0].selectbox("Stratégie", ["fixed", "martingale"])
        target = cc[1].number_input("Cash-out auto ×", 1.05, 50.0, 2.0, 0.1)
        stake = cc[2].number_input("Mise (MGA)", 70, 500000, 4000, 100)
        bankroll = cc[3].number_input("Bankroll départ", 1000, 10_000_000, 100000, 1000)
        rounds = cc[4].number_input("Manches / session", 20, 2000, 200, 10)
        if st.button("▶️ Simuler", type="primary"):
            with st.spinner("Monte-Carlo (4000 sessions)…"):
                r = S.simulate(m, strat, float(target), float(stake), float(bankroll),
                               int(rounds), sims=4000, mart_cap=float(bankroll))
            if "error" in r:
                st.error(r["error"])
            else:
                q = st.columns(4)
                q[0].metric("ROI moyen", f"{r['roi_mean_pct']:+.1f}%", f"médian {r['roi_median_pct']:+.1f}%")
                q[1].metric("Proba de PROFIT", f"{r['prob_profit_pct']:.1f}%")
                q[2].metric("Proba de RUINE", f"{r['prob_ruin_pct']:.1f}%",
                            delta_color="inverse", delta="bankroll → 0")
                q[3].metric("Drawdown moyen", f"{r['drawdown_mean_pct']:.0f}%")
                st.caption(f"P(cash-out réussi) {r['p_cashout']}% · EV théorique/manche "
                           f"{r['ev_per_round']:+.1f} MGA · sur la session {r['ev_session_theo']:+.0f} MGA "
                           f"· final probable [P5 {r['final_p5']:.0f} … P95 {r['final_p95']:.0f}]")
                st.markdown("**Trajectoire d'une session type :**")
                st.line_chart(pd.DataFrame({"bankroll": r["example_trajectory"]}))
                if strat == "martingale":
                    st.error("⚠️ La martingale double la mise après chaque perte : ROI moyen souvent "
                             "positif MAIS probabilité de ruine élevée — un seul mauvais enchaînement "
                             "vide la bankroll. C'est le piège classique.")

    with tab_c:
        st.markdown("**Comparateur de cibles** (mise fixe) — voir que l'EV ne dépend pas de la cible")
        if st.button("Comparer les cibles"):
            with st.spinner("…"):
                comp = S.compare_targets(m, targets=(1.3, 1.5, 2, 3, 5, 10),
                                         stake=float(4000), bankroll=float(100000),
                                         rounds=200, sims=3000)
            rows = [{"cible": f"{T}×", "P(cashout)": f"{r['p_cashout']}%",
                     "ROI moyen": f"{r['roi_mean_pct']:+.1f}%", "P(profit)": f"{r['prob_profit_pct']}%",
                     "P(ruine)": f"{r['prob_ruin_pct']}%", "drawdown": f"{r['drawdown_mean_pct']}%"}
                    for T, r in comp.items() if "error" not in r]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.info("L'espérance est la même partout (= −marge) ; seul le **risque** change. "
                    "Les cibles basses (1.3×) gagnent souvent mais ruinent vite ; les hautes (10×) "
                    "sont rares. Le ~2× minimise le risque de ruine — sans jamais rendre le jeu gagnant.")

    st.divider()
    st.caption("✈️ Données live via le collecteur local (session Bet261). Le crash est déterminé "
               "côté serveur AVANT tes mises (provably-fair) : aucune prédiction n'est possible. "
               "Outil d'audit et de discipline — pas de promesse de gain.")


main()
