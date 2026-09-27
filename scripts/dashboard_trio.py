"""APP CLONE — Dashboard TRIO (V2 + V5 + arbitre MARCHÉ).

Application Streamlit INDÉPENDANTE (ne touche à rien de l'existant).
Lancement : streamlit run scripts/dashboard_trio.py --server.port 8513
"""
from __future__ import annotations
import json as _j
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import re
import pandas as pd          # module-level : évite le piège UnboundLocalError si `pd`
                             # n'est importé qu'en local dans main() (bug survenu en prod)

# Marches a score exact : bannis de tout affichage (demande user). Une seule
# definition, utilisee par le bandeau "les plus probables" ET la liste ordonnee.
SCORES_EXACTS = {"Score exact", "Mi-tps CS", "2ème mi-tps - CS"}

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


def _hist_block(st, engine, home, away, leagues, n=5, show_ou35=True, n_h2h=60):
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
        h2h = _safe(_pth.head_to_head, engine, home, away, leagues, n_h2h)
        if not h2h:
            st.caption("Aucun face-à-face direct en base.")
        else:
            nz = sum(1 for m in h2h if m["tot"] == 0)
            st.caption(f"{len(h2h)} confrontations les + récentes · {nz} finies 0-0 "
                       f"({100*nz/len(h2h):.0f}%) · "
                       f"total buts moyen {sum(m['tot'] for m in h2h)/len(h2h):.1f}")
            st.caption("📊 O/U 2.5 reconstitué depuis « Total de buts » — Bet261 ne cote que la "
                       "ligne 3.5. Marge du book conservée. ✅ = issue réalisée.")
            for m in h2h[:n_h2h]:
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
                # Ligne 1 : le marché « Total de buts » en entier, ✅ sur la ligne
                # sortie — on voit d'un coup ce que le book pensait de CE score.
                # Ligne 2 : O/U 2.5 reconstitué + double chance. Deux sous-lignes car
                # tout empiler sur celle du score la rendait interminable.
                lignes = []
                tl = m.get("totals") or []
                if tl:
                    lignes.append("　📊 Total buts : " + " · ".join(
                        f"{x['label']} `{x['odd']:g}`{'✅' if x['hit'] else ''}" for x in tl))
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
                if seg:
                    lignes.append("　📊 " + " · ".join(seg))
                od25 = "".join("  " + chr(10) + l for l in lignes)
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



    # ---- 🔮 PRÉDIRE MES RENCONTRES (je choisis, il prédit — 9 ligues) ----
    with st.expander("🔮 Prédire mes rencontres — je choisis, il prédit (9 ligues)"):
        import predict_trio as _ptd
        engD = st.cache_resource(_engine)()
        st.caption("Choisis une ligue, charge les rencontres à venir, coche celles qui "
                   "t'intéressent → vainqueur, score exact, piège et value éventuels.")
        _lgn = list(LEAGUES)
        _dfi = next((i for i, k in enumerate(_lgn) if LEAGUES[k] == "InstantLeague-8060"), 0)
        sp_lg = st.selectbox("Ligue", _lgn, index=_dfi, key="sp_lg")
        sp_comp = LEAGUES[sp_lg]
        if st.button("📥 Charger les rencontres à venir", key="sp_load"):
            with _db("Chargement des rencontres…"):
                _now = datetime.now(timezone.utc)
                fx = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,
                    e.round_info rd,
                    o.odds_home oh,o.odds_draw od,o.odds_away oa,o.extra_markets xm
                    FROM events e
                    JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots
                                                   WHERE event_id=e.id)
                    LEFT JOIN results r ON r.event_id=e.id
                    WHERE r.id IS NULL AND e.expected_start IS NOT NULL
                      AND e.competition='{sp_comp}'""", engD)
                rows = []
                if len(fx):
                    fx["es"] = pd.to_datetime(fx.expected_start, utc=True)
                    fx = fx[fx.es > _now - pd.Timedelta(minutes=3)].sort_values("es").head(80)
                    for r in fx.itertuples():
                        loc = (r.es + pd.Timedelta(hours=3)).strftime("%H:%M")
                        _rdd = re.findall(r"\d+", str(r.rd or ""))
                        rows.append({"label": f"{loc} — {r.team_a} v {r.team_b}",
                                     "team_a": r.team_a, "team_b": r.team_b,
                                     "oh": float(r.oh), "od": float(r.od),
                                     "oa": float(r.oa), "xm": r.xm,
                                     "rd": int(_rdd[0]) if _rdd else None})
                st.session_state["sp_fx"] = rows
                st.session_state.pop("sp_res", None)
        fxs = st.session_state.get("sp_fx")
        if fxs is not None:
            if not fxs:
                st.info("Aucune rencontre à venir captée pour cette ligue (attends un round).")
            else:
                chos = st.multiselect(f"Tes rencontres ({len(fxs)} à venir)",
                                      [f["label"] for f in fxs], key="sp_sel")
                if st.button("🔮 Prédire ma sélection", key="sp_go", type="primary") and chos:
                    _m5 = _v2 = None
                    try:
                        with st.spinner("Fit V5+V2 (1er appel ~60-90s, puis instantané)…"):
                            _eng, _m5, _v2, _n = st.cache_resource(_fit)()
                    except Exception as exc:
                        st.error(f"Fit impossible : {exc}")
                    if _m5 is not None:
                        outs = []
                        with _db("Prédiction de ta sélection…"):
                            for f in fxs:
                                if f["label"] not in chos:
                                    continue
                                try:
                                    _m = _ptd.predict_one(
                                        engD, _m5, _v2, f["team_a"], f["team_b"],
                                        f["oh"], f["od"], f["oa"], f["xm"], lg=sp_comp)
                                except Exception as exc:
                                    _m = {"err": str(exc)}
                                try:
                                    _m["own"] = _ptd.predict_own(
                                        engD, f["team_a"], f["team_b"], lg=sp_comp,
                                        journee=f.get("rd"))
                                except Exception:
                                    _m["own"] = None
                                outs.append((f, _m))
                        st.session_state["sp_res"] = outs
        for f, m in st.session_state.get("sp_res") or []:
            st.markdown(f"#### 🕐 {f['label']}  \n`{f['oh']:g}/{f['od']:g}/{f['oa']:g}`")
            if m.get("err"):
                st.warning(f"Prédiction impossible : {m['err']}")
                continue
            # MON analyse d'abord (forme reelle, cotes non utilisees) ; le
            # marche n'est plus qu'une ligne de comparaison en dessous.
            own = m.get("own")
            if own:
                oph, opd, opa = own["x12"]
                if oph >= opd and oph >= opa:
                    o_issue, o_pi = f["team_a"], oph
                elif opa >= opd:
                    o_issue, o_pi = f["team_b"], opa
                else:
                    o_issue, o_pi = "Nul", opd
                o_top3 = " · ".join(f"{s} ({p*100:.0f}%)" for s, p in own["top3"])
                st.success(f"🧠 Mon analyse (forme virtuel Bet261) : **{o_issue}"
                           f"{' gagne' if o_issue != 'Nul' else ''}** ({o_pi*100:.0f}%) "
                           f"· score **{own['top3'][0][0]}** — Top-3 : {o_top3}")
                emo = {"V": "🟢", "N": "⚪", "D": "🔴"}
                fa = " ".join(emo.get(c, "?") for c in own.get("seq_a", ""))
                fb = " ".join(emo.get(c, "?") for c in own.get("seq_b", ""))
                st.caption(f"Forme Bet261 — {f['team_a']} : {fa} · ~{own['lam_a']} buts attendus "
                           f"| {f['team_b']} : {fb} · ~{own['lam_b']} "
                           f"({own['n_a']}/{own['n_b']} matchs virtuels, les récents pèsent plus). "
                           f"Cotes non utilisées.")
                se_a, se_b = own.get("season_a"), own.get("season_b")
                if se_a and se_b:
                    st.caption(f"📅 Saison en cours (J{own['journee']}) — "
                               f"{f['team_a']} : {se_a['v']}V {se_a['n']}N {se_a['d']}D, "
                               f"{se_a['bp']}-{se_a['bc']} buts, {se_a['pts']} pts | "
                               f"{f['team_b']} : {se_b['v']}V {se_b['n']}N {se_b['d']}D, "
                               f"{se_b['bp']}-{se_b['bc']} buts, {se_b['pts']} pts — "
                               f"fusionnée 50/50 dans le pronostic.")
            else:
                st.warning("🧠 Pas assez d'historique en base pour une analyse propre de ce duo.")
            ph, pd_, pa = m["x12"]
            if ph >= pd_ and ph >= pa:
                issue, pi, ci = f["team_a"], ph, f["oh"]
            elif pa >= pd_:
                issue, pi, ci = f["team_b"], pa, f["oa"]
            else:
                issue, pi, ci = "Nul", pd_, f["od"]
            cs = m.get("consensus_top3") or []
            t1 = m.get("top1_calibre") or (cs[0] if cs else None)
            sc = f" · score {t1[0]} ({t1[1]*100:.0f}%)" if t1 and t1[0] else ""
            st.caption(f"📊 Le marché, lui, dit : {issue}"
                       f"{' gagne' if issue != 'Nul' else ''} ({pi*100:.0f}%) "
                       f"· cote {ci:g}{sc}")
            # signaux piege + value : memes regles que la vue du round
            raisons = []
            inv = 1 / f["oh"] + 1 / f["od"] + 1 / f["oa"]
            if f["oh"] <= f["oa"]:
                o_fav, p_fav, pm_fav = f["oh"], ph, (1 / f["oh"]) / inv
            else:
                o_fav, p_fav, pm_fav = f["oa"], pa, (1 / f["oa"]) / inv
            if o_fav <= 1.7 and p_fav < pm_fav - 0.05:
                raisons.append("favori fragile")
            if pd_ >= 0.30 and o_fav <= 2.2:
                raisons.append(f"nul menaçant ({pd_*100:.0f}%)")
            conf = m.get("confidence") or 0
            if 0 < conf < 0.28:
                raisons.append("match chaotique")
            if str(m.get("accord", "")).startswith("1/"):
                raisons.append("moteurs en désaccord")
            if raisons:
                st.warning("⚠️ Piège possible : " + " · ".join(raisons))
            for team, p, o in ((f["team_a"], ph, f["oh"]), (f["team_b"], pa, f["oa"])):
                if o >= 5.0 and p * o >= 1.0:
                    st.info(f"🔦 Value repérée : **{team} gagne** — cote **{o:g}** · {p*100:.0f}%")
            st.markdown("---")

    # ---- 🧭 QUE JOUER ? — conseil tous marchés, TOUTES les rencontres ----
    with st.expander("🧭 Que jouer ? — mon conseil sur toutes les rencontres"):
        import predict_trio as _ptc2
        engC = st.cache_resource(_engine)()
        st.caption("Choisis la ligue et l'heure : j'analyse TOUS les marchés "
                   "(vainqueur, total de buts, over/under, les deux marquent, "
                   "multi-buts, score exact, minute du 1er but, 1re équipe à marquer, "
                   "mi-temps/fin de match) de CHAQUE rencontre, et je dis quoi jouer. "
                   "Prédiction issue de la seule forme des équipes.")
        c_lgs = st.multiselect("Ligues (vide = les 9)", list(LEAGUES), default=[], key="cs_lgs")
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        c_h = cc1.text_input("Heure (HH:MM Mada) — vide = les prochaines heures",
                             value="", key="cs_h", placeholder="ex: 21:03")
        c_max = cc2.number_input("Rencontres max", 1, 30, 12, 1, key="cs_max",
                                 help="Garde-fou : chaque rencontre demande une "
                                      "analyse complète de ses 11 marchés.")
        # SEUIL SUR LA COTE DU CONSEIL, demande du 27/09 : « affiche juste ceux de
        # ton pronostic >= 1,2 ».
        #
        # POURQUOI CE FILTRE A UN SENS ICI, et pourquoi il porte sur la RENCONTRE
        # et non sur les lignes de détail. Mesure sur les 12 dernières rencontres
        # en base : le conseil de tête est TOUJOURS un double chance (1X / X2 / 12)
        # — c'est mécanique, `sur` = la ligne la plus probable des 11 marchés, et
        # le double chance est structurellement le plus probable de tous. Ses cotes
        # s'étalent de 1,00 à 1,29 : cinq rencontres sur douze payaient moins de
        # 1,20 (jusqu'à 1,00, soit rien du tout). Le seuil sépare donc vraiment.
        #
        # Sur les lignes de détail, au contraire, 125 des 132 mesurées sont déjà
        # au-dessus de 1,20 : y appliquer le seuil n'aurait écarté que 7 lignes sur
        # 132. Le détail des 11 marchés reste donc ENTIER — « garder tout ».
        c_min = cc3.number_input("Cote min. du conseil", 1.00, 5.00, 1.20, 0.01,
                                 key="cs_min", format="%.2f",
                                 help="N'affiche que les rencontres dont le conseil "
                                      "de tête paie au moins ça. À 1,00 tout "
                                      "s'affiche, comme avant.")
        # QUEL conseil, demande du 27/09 : « afficher les rencontres où les
        # prédictions X2 à une cote supérieure 1,20, et les autres enlève ».
        #
        # Deux critères et non un. Sur les douze rencontres mesurées, sept
        # passent le seuil de 1,20 — mais deux seulement sont des X2 : le
        # conseil de tête est aussi souvent un 1X ou un 12, selon lequel des
        # deux camps le modèle voit le mieux tenir.
        #
        # Vide = tous les conseils, ce qui rend l'onglet à son comportement
        # d'avant sans toucher au code.
        c_sel = st.multiselect("Conseil retenu (vide = tous)",
                               ["X2", "1X", "12"], default=["X2"], key="cs_sel",
                               help="X2 = nul ou victoire extérieure. "
                                    "1X = nul ou victoire domicile. "
                                    "12 = pas de nul.")
        if st.button("🧭 Que dois-je jouer ?", key="cs_go", type="primary"):
            hh = c_h.strip()
            if hh and not re.match(r"^\d{1,2}:\d{2}$", hh):
                st.warning("Heure au format HH:MM (ex: 21:03).")
            else:
                with _db("Analyse de toutes les rencontres…"):
                    fx = _ptc2.rencontres(
                        engC, leagues=[LEAGUES[k] for k in c_lgs] or None,
                        minutes=240, heure=hh or None)
                    # Toutes les rencontres sont analysees d'un coup : plus aucune
                    # selection prealable. Le plafond evite qu'une plage large ne
                    # declenche des dizaines d'analyses completes.
                    st.session_state["cs_res"] = [
                        _ptc2.conseil(engC, r) for r in fx[:int(c_max)]]
                    st.session_state["cs_tot"] = len(fx)
        res_l = st.session_state.get("cs_res")
        if res_l is not None:
            total = st.session_state.get("cs_tot", 0)
            if not res_l:
                st.info("Aucune rencontre à venir sur ces critères — élargis les ligues, "
                        "vide l'heure, ou attends le prochain round.")
            else:
                # Le tri se fait À L'AFFICHAGE et non au calcul : bouger le seuil
                # réaffiche aussitôt, sans relancer onze analyses par rencontre.
                tri = _ptc2.filtrer_conseils(res_l, c_min, c_sel)
                gardees = tri.gardees
                n_vraies = sum(1 for r_c in gardees if not r_c.get("erreur"))
                quoi = " ou ".join(c_sel) if c_sel else "tous conseils"
                if not n_vraies:
                    st.info(f"Aucune des {len(res_l)} rencontre(s) analysée(s) ne "
                            f"porte un conseil **{quoi}** à **≥ {c_min:g}**."
                            + (f" {tri.hors_selection} conseillent autre chose."
                               if tri.hors_selection else "")
                            + (f" {tri.trop_bas} le conseillent mais sous {c_min:g}."
                               if tri.trop_bas else "")
                            + " Élargis le conseil retenu, ou baisse le seuil.")
                else:
                    st.success(
                        f"**{n_vraies} rencontre(s) retenue(s)** sur {len(res_l)} "
                        f"analysée(s)"
                        + (f" — {total} à venir en tout" if total > len(res_l) else "")
                        + f" — conseil **{quoi}** à cote **≥ {c_min:g}**."
                        + (f" {tri.hors_selection} écartée(s), autre conseil."
                           if tri.hors_selection else "")
                        + (f" {tri.trop_bas} écartée(s), conseil sous le seuil."
                           if tri.trop_bas else "")
                        + (f" {tri.sans_cote} écartée(s), conseil non coté."
                           if tri.sans_cote else ""))
                emo = {"V": "🟢", "N": "⚪", "D": "🔴"}
                for res_c in gardees:
                    if res_c.get("erreur"):
                        st.caption(f"⚠️ {res_c['erreur']}")
                        continue
                    st.markdown(f"#### `[{res_c['tag']} {res_c['local']}]` "
                                f"{res_c['home']} vs {res_c['away']}")
                    s_ = res_c.get("sur")
                    if s_:
                        cot = (f" · cote **{s_['odds']:g}**" if s_.get("odds")
                               else " · _non coté_")
                        st.success(f"**À jouer : « {s_['sel']} »**　_[{s_['marche']}]_"
                                   f"{cot} · ma proba **{s_['p']*100:.0f}%**")
                    fa = " ".join(emo.get(x, "?") for x in (res_c.get("seq_a") or ""))
                    fb = " ".join(emo.get(x, "?") for x in (res_c.get("seq_b") or ""))
                    jr = f"J{res_c['journee']} · " if res_c.get("journee") else ""
                    st.caption(f"{jr}**{res_c['attendus']} buts attendus** — "
                               f"{res_c['home']} : {fa} ~{res_c['lam_a']} · "
                               f"{res_c['away']} : {fb} ~{res_c['lam_b']}.")
                    # Detail COMPLET pour chaque rencontre : les 11 marches et,
                    # sous chacun, ses deux meilleures alternatives avec leur cote.
                    for l in res_c["lignes"]:
                        cot = f"cote **{l['odds']:g}**" if l.get("odds") else "_non coté_"
                        st.markdown(f"　• _{l['marche']}_ → **{l['sel']}** — "
                                    f"**{l['p']*100:.0f}%** · {cot}")
                        alt = " · ".join(
                            f"{t['sel']} {t['p']*100:.0f}%"
                            + (f" ({t['odds']:g})" if t.get("odds") else "")
                            for t in l["top3"][1:])
                        if alt:
                            st.caption(f"　　sinon : {alt}")
                    st.markdown("---")
                st.caption("Probas calibrées marché par marché sur 59 670 matchs "
                           "(moitié TRAIN / moitié TEST chronologique). Mon conseil "
                           "tient : annoncé 79,7% → **touché 80,1%** sur 29 835 matchs "
                           "jamais vus. Mais le book price tout : ROI ≈ **−7%**. "
                           "Le pari le plus sûr n'est pas un pari gagnant — mise petite.")


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


if __name__ == "__main__":
    main()
else:
    # exécuté par `streamlit run`
    try:
        import streamlit  # noqa
        main()
    except ModuleNotFoundError:
        pass
