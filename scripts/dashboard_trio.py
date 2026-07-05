"""APP CLONE — Dashboard TRIO (V2 + V5 + arbitre MARCHÉ).

Application Streamlit INDÉPENDANTE (ne touche à rien de l'existant).
Lancement : streamlit run scripts/dashboard_trio.py --server.port 8513
"""
from __future__ import annotations
import json as _j
import sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import re

LEAGUES = {"🏴 Angleterre": "InstantLeague-8035", "🌍 Coupe du Monde": "InstantLeague-8065",
           "🏆 Champions": "InstantLeague-8056", "🌍 CAN": "InstantLeague-8060",
           "🇮🇹 Italie": "InstantLeague-8036", "🇪🇸 Espagne": "InstantLeague-8037",
           "🇫🇷 France": "InstantLeague-8042", "🇩🇪 Allemagne": "InstantLeague-8043",
           "🇵🇹 Portugal": "InstantLeague-8044"}


def _fit():
    from scraper.config import load_settings
    from sqlalchemy import create_engine
    import predict_trio as pt
    eng = create_engine(load_settings().db_url)
    m5, v2, n = pt.fit(eng)
    return eng, m5, v2, n


def _round(models, target=None, lg="InstantLeague-8035"):
    import predict_trio as pt
    eng, m5, v2, _n = models
    return pt.predict_round(eng, m5, v2, target, lg=lg)


def _alerts():
    """Alertes de la veille : edge ligne confirmé + dérive RNG (z>3 sur 300 préd.)."""
    msgs = []
    try:
        rec = _j.loads((ROOT / "data" / "vfoot_ml" / "line_edge_history.jsonl")
                       .read_text(encoding="utf-8").strip().splitlines()[-1])
        if rec.get("confirmed"):
            msgs.append("🚨 EDGE MOUVEMENT DE LIGNE CONFIRMÉ — lance scripts/vfoot_ml/line_edge_monitor.py "
                        "pour le détail. Vérification adverse requise avant toute mise.")
    except Exception:
        pass
    try:
        fl = ROOT / "data" / "vfoot_ml" / "champion_switch.flag"
        if fl.exists():
            sw = _j.loads(fl.read_text(encoding="utf-8")).get("switched", {})
            msgs.append("🚨 BASCULE DE CHAMPION au tournoi d'algos (" +
                        ", ".join(f"{k}: {v}" for k, v in sw.items()) +
                        ") — le RNG/pricing a probablement changé de version.")
    except Exception:
        pass
    try:
        import numpy as np, pandas as pd
        from sqlalchemy import create_engine as _ce
        from scraper.config import load_settings as _ls
        d = pd.read_sql("""SELECT hit1_cal, hit3, hitx FROM trio_predictions
                           WHERE actual IS NOT NULL AND actual!='VOID'
                           ORDER BY id DESC LIMIT 300""", _ce(_ls().db_url))
        if len(d) >= 100:
            for name, obs, ceil in (("Top-1", d.hit1_cal.mean(), 0.119),
                                    ("Top-3", d.hit3.mean(), 0.316), ("1X2", d.hitx.mean(), 0.55)):
                z = (obs - ceil) / np.sqrt(ceil * (1 - ceil) / len(d))
                if abs(z) > 3:
                    msgs.append(f"⚠️ DÉRIVE RNG possible ({name} réel {obs*100:.1f}% vs plafond "
                                f"{ceil*100:.0f}%, z={z:+.1f}) — le RNG a peut-être changé de version.")
    except Exception:
        pass
    return msgs


def main():
    import streamlit as st
    st.set_page_config(page_title="TRIO — V2×V5×Marché", page_icon="⚖️", layout="wide")
    try:
        from scripts.ui_theme import inject_theme, hero
    except Exception:
        from ui_theme import inject_theme, hero
    inject_theme(st, accent="#22c55e", accent2="#2dd4bf", accent3="#38bdf8")
    hero(st, "⚖️ Prédiction TRIO",
         "V2 + V5 + arbitre Marché — trois votes à poids égaux, le marché tranche les désaccords",
         badges=["🧠 <b>V2</b> Poisson+DC", "🕐 <b>V5</b> HT/FT", "⚖️ <b>Marché</b> devigé",
                 "✅ 9 ligues", "📈 suivi forward"])

    # ---- ALERTES VEILLE (edge ligne / dérive RNG) ----
    alerts = _alerts()
    for a in alerts:
        st.error(a)
    if not alerts:
        st.caption("🟢 Veille : RAS — edge non confirmé, distribution RNG stable.")

    now_mada = datetime.now(timezone.utc) + timedelta(hours=3)
    st.metric("🕐 Heure Mada (UTC+3)", now_mada.strftime("%d/%m/%Y %H:%M"))

    # fit PARESSEUX : ne bloque plus le chargement de la page — il ne se lance
    # qu'au premier clic (spinner ~60-90s), puis reste en cache (instantané).
    cached_fit = st.cache_resource(_fit)

    cL, cA = st.columns([2, 2])
    lg_name = cL.selectbox("Ligue", list(LEAGUES), index=0)
    lg = LEAGUES[lg_name]
    auto = cA.toggle("🔄 Suivi auto (prochain round, refresh ~45s)", value=False)
    if lg != "InstantLeague-8035":
        st.caption("ℹ️ Ligue en mode MARCHÉ pur (probas dévigées, calibrées) — V2/V5 sont "
                   "entraînés sur l'anglaise.")

    cT, cB = st.columns([3, 1])
    t_str = cT.text_input("Heure Mada du round (ex: 21:03) — vide = prochain", value="", key="rt")
    go_h = cB.button("🎯 Ce round")
    go_now = st.button("🔮 Prédire le prochain round à venir")

    if go_h or go_now or auto:
        target = None
        if go_h and t_str.strip():
            d = re.findall(r"\d+", t_str)
            if len(d) >= 2:
                target = f"{int(d[0]) % 24:02d}:{int(d[1]) % 60:02d}"
        try:
            with st.spinner("Fit V5+V2 (1er appel ~60-90s, puis instantané)…"):
                models = cached_fit()
            st.caption(f"✓ V5+V2 fittés sur {models[3]} matchs (cache).")
        except Exception as exc:
            st.error(f"Fit impossible : {exc}"); return
        with st.spinner("Calcul du trio…"):
            res = _round(models, target, lg)
        if target and res.get("rounds") and target not in res["rounds"]:
            st.warning(f"Round {target} non dispo. Rounds : {res['rounds'][:10]}")
        if not res.get("matches"):
            st.info("Aucun match à venir capté (le scraper doit tourner).")
            return
        # compte à rebours jusqu'au coup d'envoi
        try:
            hh, mm = map(int, res["target"].split(":"))
            nm = datetime.now(timezone.utc) + timedelta(hours=3)
            ko = nm.replace(hour=hh, minute=mm, second=0)
            if ko < nm - timedelta(minutes=2):
                ko += timedelta(days=1)
            left = int((ko - nm).total_seconds())
            cd = f"⏳ coup d'envoi dans {max(left,0)//60}:{max(left,0)%60:02d}" if left > 0 else "🔴 en cours"
        except Exception:
            cd = ""
        st.success(f"Round {res['target']} Mada — {len(res['matches'])} matchs   {cd}")
        # ---- CADRAN DE PRÉCISION : choisis ta confiance -> meilleur pari par match ----
        import predict_trio as _ptc
        cpr1, cpr2 = st.columns([3, 2])
        want_conf = cpr1.slider("🎯 Je veux être sûr à… (%)", 50, 95, 70, 5,
                                help="Pour chaque match, l'app cherche le pari à la COTE la plus haute "
                                     "dont la probabilité atteint ce seuil. Monte le seuil = plus sûr mais "
                                     "cote plus basse ; baisse-le = plus payant mais plus risqué.") / 100.0
        cpr2.caption("La précision est un **cadran** : score exact ~31%, mais 1X2 ~55%, "
                     "O/U ~62%, Double Chance ~74%, bandes larges ~80%.")
        # ---- FILTRE CONFIANCE (prédiction sélective) ----
        matches_all = res["matches"]
        hi_only = st.toggle("🎯 Haute confiance seulement (matchs les plus prévisibles)", value=False,
                            help="Ne montre que les matchs à forte concentration Top-3 (~masse ≥0.32). "
                                 "Le Top-3 réel y grimpe à ~36-39% au lieu de 31% — MAIS ce n'est ni 100% "
                                 "ni rentable (cotes basses). Concentre ton attention, ne promet rien.")
        HI = 0.32
        shown = [m for m in matches_all if (m.get("confidence") or 0) >= HI] if hi_only else matches_all
        if hi_only:
            st.caption(f"🎯 {len(shown)}/{len(matches_all)} matchs à haute confiance ce round "
                       f"(Top-3 attendu ~36-39% vs 31% global).")
            if not shown:
                st.info("Aucun match assez concentré dans ce round — normal, ils sont rares (~10%).")
        for m in shown:
            ph, pd_, pa = m["x12"]
            conf = m.get("confidence") or 0
            badge = "🟢 haute" if conf >= 0.32 else ("🟡 moyenne" if conf >= 0.29 else "🔴 faible")
            c1, c2, c3 = st.columns([3, 2, 3])
            with c1:
                st.markdown(f"**{m['match']}**  \n`{m['cotes'][0]}/{m['cotes'][1]}/{m['cotes'][2]}`")
                st.markdown(f"1 **{ph*100:.0f}%** · X {pd_*100:.0f}% · 2 **{pa*100:.0f}%**")
                st.caption(f"confiance {badge} ({conf*100:.0f}% de masse Top-3)")
                pick = _ptc.pick_for_confidence(m.get("board") or {}, want_conf)
                if pick:
                    pmk, ps, pp, po = pick
                    st.success(f"🎯 Pour ≥{want_conf*100:.0f}% : **{ps}** [{pmk}] — "
                               f"{pp*100:.0f}% · cote {po:g}")
                else:
                    st.warning(f"Aucun pari ≥{want_conf*100:.0f}% sur ce match (baisse le seuil).")
                acc = m.get("accord", "?")
                badge = "🟢" if acc.startswith("3/") else ("🟡" if acc.startswith("2/") else "🔴")
                st.caption(f"{badge} accord moteurs : {acc}")
            with c2:
                cs = m.get("consensus_top3") or [("?", 0)]
                t1c = m.get("top1_calibre")
                if t1c:
                    st.metric("Top-1 (calibré)", t1c[0], f"{t1c[1]*100:.0f}%")
                else:
                    st.metric("CONSENSUS", cs[0][0], f"{cs[0][1]*100:.0f}%")
                st.caption("Top-3 : " + " · ".join(s for s, _ in cs[:3]))
                ov = m.get("over25_pct")
                if ov is not None:
                    st.caption(f"⚽ Over 2.5 : **{ov}%**")
            with c3:
                def line(lbl, lst):
                    return f"**{lbl}** : " + (" · ".join(f"{s}({p*100:.0f})" for s, p in lst) if lst else "—")
                st.caption(line("V2", m.get("v2_top3", [])))
                st.caption(line("V5", m.get("v5_top3", [])))
                st.caption(line("Marché", m.get("market_top3", [])))
            # ---- TOUS LES MARCHÉS du match (probas dévigées calibrées) ----
            board = m.get("board") or {}
            if board:
                with st.expander(f"📋 Tous les marchés — {m['match']} (✅ = pari probable ≥55%)"):
                    # bandeau : LES paris les plus probables du match, tous marchés confondus
                    best = sorted(((mkt, s, p, o) for mkt, rows in board.items()
                                   for s, p, o in rows if p >= 0.55),
                                  key=lambda r: -r[2])[:6]
                    if best:
                        st.markdown("🎯 **Les plus probables du match** : " + " · ".join(
                            f"**{s[:22]}** [{mkt}] {p*100:.0f}% ({o:g})" for mkt, s, p, o in best))
                        st.divider()
                    order = ["1X2", "Double Chance", "+/-", "Total de buts", "Multi-Buts", "G/NG",
                             "Pair/Impair", "Total equipe domicile", "Total equipe extérieur",
                             "G/NG equipe domicile", "G/NG equipe extérieur", "Mi-tps 1X2",
                             "Mi-tps DC", "HT/FT", "Les deux équipes marquent / 1ère mi temps",
                             "Mi-tps CS", "Score exact", "2ème mi-tps - CS", "FTTS",
                             "Minute du premier but", "1X2 & Total", "1X2 & G/NG"]
                    for mkt in order:
                        rows = board.get(mkt)
                        if not rows:
                            continue
                        top = rows[:6] if len(rows) > 8 else rows
                        st.markdown(f"**{mkt}** : " + " · ".join(
                            f"{'✅ ' if p >= 0.55 else ''}{s} **{p*100:.0f}%** ({o:g})"
                            for s, p, o in top))
                    st.caption("Probas = cotes dévigées (calibrées <2pp, prouvé sur 32k matchs). "
                               "Espérance de CHAQUE pari = −marge (~6% marchés simples, ~10-18% exotiques).")
            st.divider()

        # ================= COMBINÉ CONSEILLÉ =================
        import predict_trio as _pt
        st.subheader("🎯 Combiné conseillé — politique max-gain / min-risque")
        cc0, cc1, cc2, cc3 = st.columns(4)
        fam = cc0.selectbox("Famille", ["Sûrs (1X2/DC/OU/BTTS)", "⚽ TOTALS uniquement"], index=1)
        tgt_odds = cc1.number_input("Cote cible minimum", 1.5, 20.0, 3.0, 0.5)
        mx = int(cc2.selectbox("Jambes max", ["2", "3"], index=1))
        inter = cc3.toggle("🌍 Inter-ligues (9)", value=False,
                           help="Élargit aux matchs des 9 ligues qui démarrent dans les ~6 min")
        pool = res["matches"]
        if inter:
            try:
                pool = _pt.upcoming_all(models[0], 6) or res["matches"]
                st.caption(f"pool inter-ligues : {len(pool)} matchs des ~6 prochaines minutes")
            except Exception:
                pool = res["matches"]
        totals_mode = fam.startswith("⚽")
        if totals_mode:
            combos = _pt.build_combos(pool, float(tgt_odds), mx,
                                      markets=_pt.TOTALS_MARKETS, min_legs=1, p_min=0.20)
            st.caption("📚 Playbook totals (32k matchs, 3312 rounds backtestés) : chaque jambe coûte "
                       "sa marge (O/U 3.5 ~6%, Total exact 10-13%, Multi-Buts ~9%) — le mode totals "
                       "autorise donc les combos à **1 jambe** : un Over 3.5 seul (cote ~2.8-3.3, "
                       "ROI −5.3%) bat un triple under à même cote (ROI −18%) de 13 points.")
        else:
            combos = _pt.build_combos(pool, float(tgt_odds), mx)
        if combos:
            for i, c in enumerate(combos, 1):
                st.markdown(f"**Option {i} — cote {c['odds']:.2f} · réussite estimée {c['p']*100:.0f}%** "
                            f"· espérance {c['ev']*100:+.1f}%")
                for (mn, mkt, s, p, o) in c["legs"]:
                    st.caption(f"  • {mn} — {mkt} : **{s}** ({p*100:.0f}%, cote {o:g})")
        else:
            st.caption("Aucun combiné n'atteint la cote cible sur les marchés sûrs de ce round.")
        st.caption("⚠️ À cote cible fixée, on te donne le combiné le PLUS PROBABLE (marchés à marge "
                   "fine uniquement : 1X2, Double Chance, O/U 3.5, G/NG ; indépendance inter-matchs "
                   "prouvée). Aucun pari n'a d'espérance positive — c'est la façon la moins risquée "
                   "d'atteindre la cote visée, pas une promesse de gain.")

        # ================= ONGLETS SPÉCIALISÉS PAR MARCHÉ =================
        st.subheader("📊 Vues spécialisées par marché")
        t_ou, t_htft, t_tot, t_gng = st.tabs(
            ["⬆⬇ Over/Under 3.5", "🕐 HT/FT", "🔢 Total exact", "⚽ G/NG"])

        def _family_view(tab, keys, marge_note, show_ov25=False, per_match_n=7):
            with tab:
                picks = []
                for m in res["matches"]:
                    b = m.get("board") or {}
                    rows_all = [(mk, s, p, o) for mk in keys for s, p, o in b.get(mk, [])]
                    if not rows_all:
                        continue
                    rows_all.sort(key=lambda r: -r[2])
                    extra = (f"  ·  Ov2.5 calibré **{m['over25_pct']}%**"
                             if show_ov25 and m.get("over25_pct") is not None else "")
                    st.markdown(f"**{m['match']}** : " + " · ".join(
                        f"{s} **{p*100:.0f}%** ({o:g})" for _mk, s, p, o in rows_all[:per_match_n])
                        + extra)
                    picks += [(m["match"], mk, s, p, o) for mk, s, p, o in rows_all]
                if not picks:
                    st.caption("Marché non coté sur ce round."); return
                st.divider()
                st.markdown("**💡 Recommandations du round** (les plus probables)")
                picks.sort(key=lambda r: -r[3])
                for i, (mn, mk, s, p, o) in enumerate(picks[:5], 1):
                    st.markdown(f"{i}. {'✅ ' if p >= 0.55 else ''}**{mn}** — {s} [{mk}] : "
                                f"**{p*100:.0f}%** (cote {o:g})")
                for tg in (2.0, 3.0):
                    cs = _pt.build_combos(res["matches"], tg, 3, top=1,
                                          markets=set(keys), min_legs=1, p_min=0.15)
                    if cs:
                        c = cs[0]
                        legs_txt = "  +  ".join(f"{l[0]} · {l[2]} ({l[3]*100:.0f}%)" for l in c["legs"])
                        st.markdown(f"**Meilleure voie vers cote ≥{tg:g}** → cote {c['odds']:.2f}, "
                                    f"réussite {c['p']*100:.0f}% : {legs_txt}")
                st.caption(marge_note)

        _family_view(t_ou, ["+/-"],
                     "Marge ~6%/pari — le MEILLEUR marché totals (cf. playbook). "
                     "Ov2.5 calibré = notre proba maison (odds→sim→calibration).",
                     show_ov25=True, per_match_n=2)
        _family_view(t_htft, ["HT/FT", "Mi-tps 1X2"],
                     "Marge : Mi-tps 1X2 ~7.7%, HT/FT ~11% — dimension temps calibrée "
                     "(prouvé campagne 17). Le plus probable est presque toujours 1/1 ou X/X.",
                     per_match_n=5)
        _family_view(t_tot, ["Total de buts"],
                     "Marge ~10-12.7% — le total le plus fréquent est 3 (25.6% des matchs). "
                     "⚠️ préfère l'onglet O/U 3.5 : même famille, moitié moins cher.",
                     per_match_n=7)
        _family_view(t_gng, ["G/NG", "Les deux équipes marquent / 1ère mi temps"],
                     "Marge ~5.6% (G/NG plein temps) — 2e meilleur marché après O/U. "
                     "BTTS 1ère mi-temps affiché en complément (~24% de oui).",
                     per_match_n=4)

    # ---- SUIVI FORWARD RÉEL (rempli par scripts/trio_tracker.py) ----
    st.divider()
    st.subheader("📈 Suivi réel (forward)")
    try:
        import pandas as pd
        from sqlalchemy import create_engine as _ce
        from scraper.config import load_settings as _ls
        _eng = _ce(_ls().db_url)
        trk = pd.read_sql("""SELECT hit1, hit1_cal, hit3, hitx FROM trio_predictions
                             WHERE actual IS NOT NULL AND actual != 'VOID'
                             ORDER BY id DESC LIMIT 500""", _eng)
        if len(trk):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Top-1 calibré", f"{100*trk.hit1_cal.mean():.1f}%", f"n={len(trk)} · plafond ~11.9%")
            k2.metric("Top-1 brut", f"{100*trk.hit1.mean():.1f}%", "plafond ~11.7%")
            k3.metric("Top-3", f"{100*trk.hit3.mean():.1f}%", "plafond ~31.6%")
            k4.metric("1X2", f"{100*trk.hitx.mean():.1f}%", "plafond ~55%")
            st.caption("Prédictions figées AVANT le coup d'envoi puis scorées au résultat "
                       "(scripts/trio_tracker.py). La seule mesure honnête.")
        else:
            st.caption("Pas encore de prédictions scorées — le tracker (trio_tracker.py) accumule.")
        # ---- suivi des COMBINÉS conseillés (annoncé vs réel, par famille) ----
        try:
            cb = pd.read_sql("""SELECT COALESCE(family,'safe') family, p_est, odds, won, pnl
                                FROM combo_suggestions WHERE won >= 0
                                ORDER BY id DESC LIMIT 1000""", _eng)
            if len(cb):
                st.markdown("**🎯 Combinés conseillés (cote ≥3, figés avant coup d'envoi) :**")
                for famname, g in cb.groupby("family"):
                    lbl = "⚽ TOTALS" if famname == "totals" else "Sûrs"
                    q1, q2, q3 = st.columns(3)
                    q1.metric(f"{lbl} — réussite réelle", f"{100*g.won.mean():.1f}%",
                              f"annoncée {100*g.p_est.mean():.1f}% · n={len(g)}")
                    q2.metric("ROI cumulé", f"{100*g.pnl.mean():+.1f}%")
                    q3.metric("Cote moyenne", f"{g.odds.mean():.2f}")
            else:
                st.caption("Combinés conseillés : le tracker fige 1 combiné sûr + 1 combiné totals "
                           "par round — stats dès les premiers règlements.")
        except Exception:
            pass
    except Exception:
        st.caption("Suivi indisponible (lancer scripts/trio_tracker.py au moins une fois).")

    st.info("⚠️ RNG calibré, pas d'edge directionnel prouvé — le trio améliore la ROBUSTESSE (arbitrage des "
            "désaccords), pas le plafond de précision.")

    # ---- SUIVI AUTO : re-prédit le prochain round toutes les ~45 s ----
    if auto:
        time.sleep(45)
        st.rerun()


if __name__ == "__main__":
    main()
else:
    # exécuté par `streamlit run`
    try:
        import streamlit  # noqa
        main()
    except ModuleNotFoundError:
        pass
