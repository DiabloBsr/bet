"""APP CLONE — Dashboard TRIO (V2 + V5 + arbitre MARCHÉ).

Application Streamlit INDÉPENDANTE (ne touche à rien de l'existant).
Lancement : streamlit run scripts/dashboard_trio.py --server.port 8513
"""
from __future__ import annotations
import json as _j
import sys, time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import re
import pandas as pd          # module-level : évite le piège UnboundLocalError si `pd`
                             # n'est importé qu'en local dans main() (bug survenu en prod)

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


def _engine():
    """Engine seul (instantané) — pour le scanner cross-ligues (pas de fit requis).
    timeout=30s : encaisse les locks SQLite quand le scraper écrit en parallèle."""
    from scraper.config import load_settings
    from sqlalchemy import create_engine
    return create_engine(load_settings().db_url, connect_args={"timeout": 30})


@contextmanager
def _db(label: str):
    """Encadre un accès base derrière un spinner, et rate proprement.

    Le scraper écrit en continu dans la même base SQLite : rencontrer un verrou
    est un fonctionnement NORMAL, pas un bug. Sans cette garde, une base occupée
    affiche une trace Python en pleine page — la classe de bug qui a provoqué la
    boucle de crash sur Hugging Face. On explique, puis on arrête le rendu au lieu
    de laisser la suite planter sur une variable jamais affectée.
    """
    import streamlit as st
    try:
        with st.spinner(label):   # surtout PAS _db(label) : la garde s'appelait
            yield                 # elle-meme -> RecursionError sur chaque acces base.
    except Exception as exc:
        m = str(exc).lower()
        if "locked" in m or "busy" in m or "timeout" in m:
            st.warning("⏳ Base occupée par le scraper (écriture en cours) — "
                       "relance dans quelques secondes.")
        elif "malformed" in m or "corrupt" in m:
            st.error("💥 Lecture incohérente (la base était en cours d'écriture) — relance.")
        else:
            st.error(f"❌ Échec : {type(exc).__name__} — {exc}")
        st.stop()


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
        h = (ROOT / "data" / "vfoot_ml" / "seeded_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        rec = _j.loads(h[-1])
        if rec.get("confirmed"):
            msgs.append(f"🚨 CYCLE SEEDÉ CONFIRMÉ (théorie en ligne #1) — après 5 unders, ROI OOS "
                        f"{100*rec.get('roi_oos',0):+.1f}% IC95 au-dessus de 0. Vérif adverse avant toute mise.")
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


def _hist_block(st, engine, home, away, leagues, n=5, show_ou35=True):
    """Composant historique réutilisable : 3 menus (H2H / équipe home / équipe away),
    du + récent au + ancien. Utilisable partout dans l'app sur les 9 ligues."""
    import predict_trio as _pth

    def _row(m):
        emo = "🟢" if m["res"] == "V" else ("⚪" if m["res"] == "N" else "🔴")
        return (f"{emo} `{m['date']}` · {m['side']} vs **{m['opp']}** — **{m['gf']}-{m['ga']}** "
                f"({m['tot']} but{'s' if m['tot'] != 1 else ''}, cote {m['odds']:g})")
    def _safe(fn, *a):
        """Ici on dégrade au lieu d'arrêter : les 3 onglets sont indépendants,
        l'un peut échouer sur un verrou sans priver l'utilisateur des deux autres."""
        try:
            return fn(*a)
        except Exception as exc:
            st.caption(f"⏳ Historique indisponible (base occupée) : {type(exc).__name__}")
            return []

    t1, t2, t3 = st.tabs([f"⚔️ Face-à-face", f"🏠 {home}", f"✈️ {away}"])
    with t1:
        h2h = _safe(_pth.head_to_head, engine, home, away, leagues)
        if not h2h:
            st.caption("Aucun face-à-face direct en base.")
        else:
            nz = sum(1 for m in h2h if m["tot"] == 0)
            st.caption(f"{len(h2h)} confrontations les + récentes · {nz} finies 0-0 "
                       f"({100*nz/len(h2h):.0f}%) · "
                       f"total buts moyen {sum(m['tot'] for m in h2h)/len(h2h):.1f}")
            st.caption("📊 O/U 2.5 reconstitué depuis « Total de buts » — Bet261 ne cote que la "
                       "ligne 3.5. Marge du book conservée. ✅ = issue réalisée.")
            for m in h2h[:30]:
                mark = " 🥅" if m["tot"] == 0 else ""
                ch = f" `{m['oh']:g}`" if m.get("oh") else ""
                ca = f" `{m['oa']:g}`" if m.get("oa") else ""
                cx = f" · nul `{m['od']:g}`" if m.get("od") else ""
                ov, un = m.get("o_over35"), m.get("o_under35")
                ou = ""
                if show_ou35 and (ov or un):    # ✅ = le côté O/U 3.5 réellement sorti (total ≥4 = over)
                    hit_over = m["tot"] >= 4
                    parts = []
                    if ov:
                        parts.append(f"O3.5 `{ov:g}`{'✅' if hit_over else ''}")
                    if un:
                        parts.append(f"U3.5 `{un:g}`{'✅' if not hit_over else ''}")
                    ou = " · " + " / ".join(parts)
                jr = f"`J{m['journee']}` " if m.get("journee") else ""
                hmin, amin = m.get("home_min") or [], m.get("away_min") or []
                gm = ""
                if hmin or amin:                # minutes des buts (dom / ext)
                    dm = " ".join(f"{x}'" for x in hmin) or "—"
                    xm2 = " ".join(f"{x}'" for x in amin) or "—"
                    gm = f"  \n　⚽ {m['home']} : {dm}  ·  {m['away']} : {xm2}"
                # O/U 2.5 reconstitué + double chance coté, sur leur propre sous-ligne :
                # la ligne principale porte déjà 1X2 et O/U 3.5, tout y empiler la rendrait
                # illisible. ✅ marque l'issue réellement sortie (over 2.5 = total >= 3).
                seg = []
                over25 = m["tot"] >= 3
                p25 = []
                if m.get("o_over25"):
                    p25.append(f"O2.5 `{m['o_over25']:g}`{'✅' if over25 else ''}")
                if m.get("o_under25"):
                    p25.append(f"U2.5 `{m['o_under25']:g}`{'✅' if not over25 else ''}")
                if p25:
                    seg.append(" / ".join(p25))
                pdc = []
                if m.get("dc_1x"):
                    pdc.append(f"1X `{m['dc_1x']:g}`{'✅' if m['sa'] >= m['sb'] else ''}")
                if m.get("dc_x2"):
                    pdc.append(f"X2 `{m['dc_x2']:g}`{'✅' if m['sb'] >= m['sa'] else ''}")
                if m.get("dc_12"):
                    pdc.append(f"12 `{m['dc_12']:g}`{'✅' if m['sa'] != m['sb'] else ''}")
                if pdc:
                    seg.append("DC " + " / ".join(pdc))
                od25 = "  " + chr(10) + "　📊 " + " · ".join(seg) if seg else ""
                st.markdown(f"{jr}`{m['date']}` — {m['home']}{ch} **{m['sa']}-{m['sb']}** "
                            f"{ca}{m['away']}{cx}{mark}{ou}{od25}{gm}")
    with t2:
        hh = _safe(_pth.match_history, engine, home, n, leagues)
        if not hh:
            st.caption("Pas d'historique.")
        for m in hh:
            st.markdown(_row(m))
    with t3:
        ha = _safe(_pth.match_history, engine, away, n, leagues)
        if not ha:
            st.caption("Pas d'historique.")
        for m in ha:
            st.markdown(_row(m))




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



    # ---- 🔦 DÉBUSQUEUR GROSSES CÔTES + HISTORIQUE ----
    with st.expander("🔦 Débusqueur grosses cotes + historique (9 ligues)"):
        import predict_trio as _ptd
        engD = st.cache_resource(_engine)()
        st.caption("Trouve les matchs à venir avec une sélection à GROSSE COTE (n'importe quel marché), triés par chance réelle.")
        db1, db2, db3 = st.columns([3, 2, 2])
        _lgn = list(LEAGUES)
        _dfi = next((i for i, k in enumerate(_lgn) if LEAGUES[k] == "InstantLeague-8060"), 0)
        db_lg = db1.selectbox("Ligue", _lgn, index=_dfi, key="db_lg")
        db_comp = LEAGUES[db_lg]
        db_lo = db2.number_input("Cote min", 2.0, 100.0, 8.0, 0.5, key="db_lo")
        db_hi = db3.number_input("Cote max", 2.0, 200.0, 50.0, 1.0, key="db_hi")
        db_mkts = st.multiselect("Marchés à scanner", _ptd.BIG_ODDS_MARKETS,
                                 default=["1X2", "Total de buts", "+/-", "G/NG"], key="db_mkts")
        dbt1, dbt2 = st.columns(2)
        db_ws = dbt1.text_input("De (HH:MM Mada — vide = maintenant)", value="", key="db_ws", placeholder="ex: 21:00")
        db_we = dbt2.text_input("À (HH:MM Mada)", value="", key="db_we", placeholder="ex: 22:00")
        if st.button("🔦 Débusquer", key="db_go", type="primary"):
            sl, el = db_ws.strip(), db_we.strip()
            valid = re.compile(r"^\d{1,2}:\d{2}$")
            if (sl and not valid.match(sl)) or (el and not valid.match(el)) or (bool(sl) != bool(el)):
                st.warning("Heures au format HH:MM, les deux ou aucune.")
            else:
                sl2 = sl.zfill(5) if sl else None
                el2 = el.zfill(5) if el else None
                with _db("Scan des grosses cotes + historiques…"):
                    st.session_state["db_res"] = _ptd.big_odds_fixtures(
                        engD, [db_comp], min_odds=float(db_lo), max_odds=float(db_hi),
                        markets=db_mkts or None, start_local=sl2, end_local=el2,
                        top=15, with_context=True, ctx_n=5)
        res = st.session_state.get("db_res")
        if res is not None:
            if not res:
                st.info("Aucune grosse cote dans ces critères (élargis la bande/les marchés ou attends un round).")
            else:
                st.success(f"{len(res)} grosses cotes — forme des équipes + face-à-face à côté, la + probable d'abord :")

                def _forme(hist):
                    if not hist:
                        return "_(pas d'historique)_"
                    emo = {"V": "🟢", "N": "⚪", "D": "🔴"}
                    seq = " ".join(emo.get(m["res"], "?") for m in hist)
                    sco = " · ".join(f"{m['gf']}-{m['ga']}" for m in hist)
                    return f"{seq}  ({sco})"
                for m in res:
                    st.markdown(f"#### 🎲 {m['local']} · {m['home']} vs {m['away']}")
                    st.markdown(f"**{m['sel']}** `[{m['market']}]` — cote **{m['odds']:g}** · "
                                f"**{m['p']*100:.0f}%** de chance")
                    st.markdown(f"　🏠 **{m['home']}** : {_forme(m.get('home_hist'))}")
                    st.markdown(f"　✈️ **{m['away']}** : {_forme(m.get('away_hist'))}")
                    if m.get("h2h_n"):
                        rec = " · ".join(f"{x['sa']}-{x['sb']}" for x in m.get("h2h_recent", []))
                        st.markdown(f"　⚔️ **Face-à-face** : {m['h2h_n']} matchs · "
                                    f"{m['h2h_zeros']}× 0-0 ({100*m['h2h_zeros']/m['h2h_n']:.0f}%) · "
                                    f"{m['h2h_avg']} buts/m · récents : {rec}")
                    else:
                        st.markdown("　⚔️ Face-à-face : aucun en base")
                    st.markdown("---")
                st.caption('🟢 victoire · ⚪ nul · 🔴 défaite (du + récent à gauche).')

    # ---- 🔎 HISTORIQUE & FACE-À-FACE (choix manuel, 9 ligues) ----
    with st.expander("🔎 Historique & face-à-face — deux équipes au choix (9 ligues)"):
        import predict_trio as _pth2
        engH = st.cache_resource(_engine)()
        st.caption('Choisis une ligue et deux équipes → face-à-face direct + 5 derniers matchs de chacune (du + récent au + ancien).')
        hl1, hl2, hl3 = st.columns([2, 2, 2])
        h_lg = hl1.selectbox("Ligue", list(LEAGUES), index=_dfi, key="h_lg")
        h_comp = LEAGUES[h_lg]
        _hteams = _pth2.league_teams(engH, h_comp)
        if _hteams:
            h_home = hl2.selectbox("Équipe A (domicile)", _hteams, index=0, key="h_home")
            h_away = hl3.selectbox("Équipe B (extérieur)", _hteams,
                                   index=min(1, len(_hteams)-1), key="h_away")
            h_ou35 = st.checkbox("Afficher les cotes Under / Over 3.5", value=True, key="h_ou35")
            if st.button("🔎 Afficher l'historique", key="h_go", type="primary"):
                if h_home == h_away:
                    st.warning("Choisis deux équipes différentes.")
                else:
                    _hist_block(st, engH, h_home, h_away, [h_comp], n=5, show_ou35=h_ou35)
        else:
            st.info("Pas d'équipes trouvées pour cette ligue.")

    # ---- 🎯 DÉBUSQUEUR UNDER 3.5 (9 ligues, cote cible, ordre du round) ----
    with st.expander("🎯 Débusqueur Under 3.5 — 9 ligues, triés par round"):
        import predict_trio as _ptu35
        engU = st.cache_resource(_engine)()
        st.caption("Matchs à venir des 9 ligues dont la cote Under 3.5 vaut ~ta cible (1.68 par "
                   "défaut), dans l'ordre des rounds. Filtre horaire optionnel (De… À…).")
        u1, u2 = st.columns(2)
        u_target = u1.number_input("Cote Under 3.5 (exacte)", 1.05, 3.0, 1.68, 0.01, key="u35_t")
        u_tol = u2.number_input("Tolérance ±", 0.0, 1.0, 0.03, 0.01, key="u35_tol",
                                help="0.03 ≈ exactement la cible. Monte-la pour élargir.")
        u3, u4 = st.columns(2)
        u_ws = u3.text_input("De (HH:MM Mada — vide = maintenant)", value="", key="u35_ws",
                             placeholder="ex: 21:00")
        u_we = u4.text_input("À (HH:MM Mada)", value="", key="u35_we", placeholder="ex: 22:00")
        if st.button("🎯 Débusquer Under 3.5", key="u35_go", type="primary"):
            sl, el = u_ws.strip(), u_we.strip()
            valid = re.compile(r"^\d{1,2}:\d{2}$")
            if (sl and not valid.match(sl)) or (el and not valid.match(el)) or (bool(sl) != bool(el)):
                st.warning("Heures au format HH:MM, les deux ou aucune.")
            else:
                sl2 = sl.zfill(5) if sl else None
                el2 = el.zfill(5) if el else None
                with _db("Scan Under 3.5 — 9 ligues…"):
                    st.session_state["u35_res"] = _ptu35.under35_scan(
                        engU, target=float(u_target), tol=float(u_tol),
                        start_local=sl2, end_local=el2)
        u_res = st.session_state.get("u35_res")
        if u_res is not None:
            if not u_res:
                st.info("Aucun match Under 3.5 ~ cible pour l'instant (élargis la tolérance ou attends un round).")
            else:
                if u_res[0].get("recent"):
                    st.warning("⏳ Aucun match à venir capté — voici les derniers matchs réels (exemples).")
                st.success(f"{len(u_res)} matchs Under 3.5 ≈ {u_target:g} (±{u_tol:g}), dans l'ordre des rounds :")
                for m in u_res[:60]:
                    o35 = f" · O3.5 `{m['over35']:g}`" if m.get("over35") else ""
                    tag = " · *(passé)*" if m.get("recent") else ""
                    st.markdown(f"**{m['local']}** `[{m['tag']}]` {m['home']} vs {m['away']} — "
                                f"**U3.5 `{m['under35']:g}`**{o35}{tag}")
                st.caption("Trié par heure de round. Rappel : Under 3.5 reste −EV (la cote paie la proba). Indicateur.")

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

    # ---- Confiance : À CHOISIR AVANT de prédire (persistée, n'efface pas le round) ----
    cpr1, cpr2 = st.columns([3, 2])
    want_conf = cpr1.slider("🎯 Je veux être sûr à… (%)", 50, 95, 70, 5, key="want_conf",
                            help="Pour chaque match, l'app cherche le pari à la COTE la plus haute "
                                 "dont la probabilité atteint ce seuil. Monte = plus sûr / cote plus "
                                 "basse ; baisse = plus payant / plus risqué.") / 100.0
    hi_only = cpr2.toggle("🎯 Haute confiance seulement", value=False, key="hi_only",
                          help="Ne montre que les matchs à forte concentration Top-3 (~masse ≥0.32).")

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
        except Exception as exc:
            st.error(f"Fit impossible : {exc}"); return
        with _db("Calcul du trio…"):
            res_new = _round(models, target, lg)
        if target and res_new.get("rounds") and target not in res_new["rounds"]:
            st.warning(f"Round {target} non dispo. Rounds : {res_new['rounds'][:10]}")
        st.session_state["pred_res"] = res_new

    res = st.session_state.get("pred_res")
    if res is not None and not res.get("matches"):
        st.info("Aucun match à venir capté (le scraper doit tourner).")
    if res and res.get("matches"):
        import predict_trio as _ptc
        _pt = _ptc
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
        st.caption("La précision est un cadran : score exact ~31%, 1X2 ~55%, O/U ~62%, "
                   "Double Chance ~74%, bandes larges ~80%.")
        matches_all = res["matches"]
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

        # ================= PRONOSTIC — grosses cotes value du round (max 2 matchs) =================
        st.subheader("🔦 Pronostic — grosses cotes value du round")
        best = []
        for m in res["matches"]:
            # on exclut les cotes PLAFONNÉES (≥90) : leur value apparente est un artefact
            # du plafond du site (le pire pari réel, prouvé −57%).
            sels = [(mkt, s, p, o) for mkt, rows in (m.get("board") or {}).items()
                    for s, p, o in rows if o < 90.0]
            # une VRAIE grosse cote (≥5, jusqu'à 10/20/50…) sinon ≥3 ; la PLUS PROBABLE
            grosses = [x for x in sels if x[3] >= 5.0] or [x for x in sels if x[3] >= 3.0]
            if not grosses:
                continue
            mkt, s, p, o = max(grosses, key=lambda x: x[2])
            best.append((m["match"], mkt, s, p, o))
        best.sort(key=lambda r: -r[3])
        if best:
            for i, (mn, mk, s, p, o) in enumerate(best[:2], 1):
                st.markdown(f"**{i}. {mn}** → **{s}** `[{mk}]` · cote **{o:g}** · **{p*100:.0f}%**")
            st.caption("La grosse cote la plus probable de chaque match (2 max) — même 10, 20+ si "
                       "elle sort du lot. Reste −EV (la cote paie déjà). Indicateur, pas une mise.")
        else:
            st.caption("Aucune cote exploitable dans ce round.")

    # ---- SUIVI FORWARD RÉEL (rempli par scripts/trio_tracker.py) ----
    st.divider()
    st.subheader("📈 Suivi réel (forward)")
    try:
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
