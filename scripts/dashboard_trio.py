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

    # ---- ⚽ OVER / UNDER 2,5 : MES DEUX PRONOSTICS LES PLUS SÛRS ----
    with st.expander("⚽ Over / Under 2,5 — mes deux pronostics les plus sûrs"):
        import predict_trio as _pto25
        engO = st.cache_resource(_engine)()
        st.caption("Je prédis sur la SEULE forme des équipes dans le virtuel Bet261 "
                   "(Poisson attaque/défense) et je te donne les deux matchs où je suis "
                   "le plus sûr : un pour le +2,5, un pour le −2,5. Les cotes affichées "
                   "sont celles de l'app, à titre d'exécution.")
        o_lgs = st.multiselect("Ligues à analyser (vide = les 9)", list(LEAGUES),
                               default=[], key="o25_lgs")
        oc1, oc2 = st.columns(2)
        o_h = oc1.text_input("Heure du match (HH:MM Mada) — vide = tous",
                             value="", key="o25_h", placeholder="ex: 21:03")
        o_c = oc2.text_input("Cote du match — vide = toutes", value="", key="o25_c",
                             placeholder="ex: 1.41",
                             help="Une cote que tu vois sur le match dans Bet261 "
                                  "(1X2, Total de buts, Multi-Buts, +/-) : elle sert "
                                  "à retrouver ce match précis.")
        if st.button("🔮 Mes deux pronostics", key="o25_go", type="primary"):
            hh, cc = o_h.strip(), o_c.replace(",", ".").strip()
            if hh and not re.match(r"^\d{1,2}:\d{2}$", hh):
                st.warning("Heure au format HH:MM (ex: 21:03).")
            elif cc and not re.match(r"^\d+(\.\d+)?$", cc):
                st.warning("Cote au format numérique (ex: 1.41).")
            else:
                with _db("Analyse des matchs à venir…"):
                    st.session_state["o25_res"] = _pto25.ou25_picks(
                        engO, leagues=[LEAGUES[k] for k in o_lgs] or None, minutes=240,
                        heure=hh or None, cote=float(cc) if cc else None)
                st.session_state["o25_crit"] = (hh, cc)
        o_res = st.session_state.get("o25_res")
        if o_res is not None:
            if not o_res.get("over") and not o_res.get("under"):
                hh, cc = st.session_state.get("o25_crit", ("", ""))
                crit = []
                if hh:
                    crit.append(f"à **{hh}**")
                if cc:
                    crit.append(f"portant la cote **{cc}**")
                if crit:
                    st.info("Aucun match " + " et ".join(crit) + " parmi les prochaines "
                            "heures. Vérifie l'heure Mada du round, élargis les ligues, "
                            "ou vide ces deux champs pour voir tous les matchs.")
                else:
                    st.info("Aucun match à venir analysable — ajoute des ligues, "
                            "ou attends le prochain round.")
            else:
                emo = {"V": "🟢", "N": "⚪", "D": "🔴"}

                def _rendre(m, sens, titre, phrase):
                    if not m:
                        st.info(f"Pas de pronostic {sens} disponible pour l'instant.")
                        return
                    st.markdown(f"### {titre}")
                    st.success(f"**`[{m['tag']} {m['local']}]` {m['home']} vs {m['away']}** → "
                               f"**{phrase}** · ma proba **{m['p_' + sens]*100:.0f}%**")
                    fa = " ".join(emo.get(c, "?") for c in (m.get("seq_a") or ""))
                    fb = " ".join(emo.get(c, "?") for c in (m.get("seq_b") or ""))
                    st.caption(f"**{m['attendus']} buts attendus** — {m['home']} : {fa} "
                               f"~{m['lam_a']} · {m['away']} : {fb} ~{m['lam_b']}.")
                    d, eq = m.get("direct"), m.get("equivalent")
                    if d:
                        mieux = ""
                        if eq and d["odds"] > eq:
                            mieux = (f"　— et c'est mieux que répartir la mise sur "
                                     f"0/1/2, qui n'équivaut qu'à {eq:g}")
                        st.info(f"✅ **En un seul clic sur Bet261** : « {d['sel']} » "
                                f"_[{d['marche']}]_ → cote **{d['odds']:g}**{mieux}")
                    elif m.get("cellules"):
                        st.warning("⚠️ Bet261 ne cote pas ce sens directement : il faut "
                                   "répartir la mise sur « Total de buts ».")
                    if m.get("cellules"):
                        lignes = "　·　".join(f"**{c['total']}** → **{c['odds']:g}** "
                                             f"({c['part']:.0f}%)" for c in m["cellules"])
                        st.markdown(f"　« Total de buts » : {lignes}")
                        if eq:
                            st.caption(f"↳ ces paris réunis équivalent à une cote de **{eq:g}** "
                                       f"(calcul — ce chiffre n'apparaît pas dans Bet261).")
                    if m.get("voisins"):
                        v = "　·　".join(f"« {x['sel']} » _[{x['marche']}]_ **{x['odds']:g}**"
                                        for x in m["voisins"])
                        st.caption(f"Paris voisins : {v}")

                _rendre(o_res.get("over"), "over", "🔼 Mon Over 2,5 le plus sûr",
                        "plus de 2,5 buts")
                st.markdown("---")
                _rendre(o_res.get("under"), "under", "🔽 Mon Under 2,5 le plus sûr",
                        "moins de 2,5 buts")
                st.caption("Fiabilité mesurée sur 59 670 matchs (moitié TRAIN / moitié TEST "
                           "chronologique, table isotone) : mes **Over** les plus sûrs "
                           "touchent ~**75%** (pour 66-68% annoncé) contre 54% en moyenne ; "
                           "mes **Under** les plus sûrs ~**77%** là où le pari existe "
                           "vraiment, contre 46% en moyenne. J'annonce volontairement moins "
                           "que le mesuré. Mais le book price tout : ROI ≈ **−8 à −10%** "
                           "dans les deux sens — plus sûr ne veut pas dire gagnant, "
                           "mise petite.")


    # ---- 🥅 TOTAL DE BUTS — MON TOP 3 (analyse propre, ligues au choix) ----
    with st.expander("🥅 Total de buts — mon top 3 le plus sûr (analyse propre)"):
        import predict_trio as _pttg
        engT = st.cache_resource(_engine)()
        st.caption("Je prédis le NOMBRE EXACT de buts de chaque match à venir à partir "
                   "de la SEULE forme des équipes dans le virtuel Bet261 (Poisson sur "
                   "attaque/défense). Les cotes n'entrent ni dans la prédiction ni dans "
                   "le classement : elles sont affichées après coup, pour information.")
        t_lgs = st.multiselect("Ligues à analyser (vide = les 9)", list(LEAGUES),
                               default=[], key="tg_lgs",
                               help="Plus tu ouvres large, meilleur est le tri : "
                                    "le top se détache d'autant mieux que le vivier est grand.")
        tg1, tg2 = st.columns(2)
        t_top = tg1.number_input("Combien de matchs", 1, 10, 3, 1, key="tg_top")
        t_win = tg2.number_input("Fenêtre (minutes à venir)", 30, 720, 180, 30, key="tg_win")
        tw1, tw2 = st.columns(2)
        t_ws = tw1.text_input("De (HH:MM Mada — vide = maintenant)", value="",
                              key="tg_ws", placeholder="ex: 21:00")
        t_we = tw2.text_input("À (HH:MM Mada)", value="", key="tg_we", placeholder="ex: 22:00")
        if st.button("🥅 Prédire les totaux", key="tg_go", type="primary"):
            sl, el = t_ws.strip(), t_we.strip()
            valid = re.compile(r"^\d{1,2}:\d{2}$")
            if (sl and not valid.match(sl)) or (el and not valid.match(el)) or (bool(sl) != bool(el)):
                st.warning("Heures au format HH:MM, les deux ou aucune.")
            else:
                with _db("Analyse du total de buts de chaque match…"):
                    st.session_state["tg_res"] = _pttg.totals_scan(
                        engT, leagues=[LEAGUES[k] for k in t_lgs] or None,
                        minutes=int(t_win),
                        start_local=sl.zfill(5) if sl else None,
                        end_local=el.zfill(5) if el else None,
                        top=int(t_top))
        t_res = st.session_state.get("tg_res")
        if t_res is not None:
            if not t_res:
                st.info("Aucun match à venir analysable (élargis la fenêtre, ajoute des "
                        "ligues, ou attends un round).")
            else:
                st.markdown(f"### 🎯 Mon top {len(t_res)} — total de buts")
                for i, m in enumerate(t_res, 1):
                    _c = (f"· cote **{m['odds']:g}** " if m.get("odds")
                          else "· _non coté par le book_ ")
                    ligne = (f"**{i}. `[{m['tag']} {m['local']}]` {m['home']} vs {m['away']}** → "
                             f"**{m['label']} but{'s' if m['total'] != 1 else ''}** "
                             f"{_c}· ma proba **{m['p_mine_cal']*100:.0f}%**")
                    (st.success if i == 1 else st.markdown)(ligne)
                    alt = " · ".join(f"{t['total']} ({t['p']*100:.0f}%"
                                     + (f", cote {t['odds']:g})" if t['odds'] else ")")
                                     for t in m["top3"])
                    st.caption(f"Mes 3 totaux les plus probables : {alt}")
                    emo = {"V": "🟢", "N": "⚪", "D": "🔴"}
                    fa = " ".join(emo.get(c, "?") for c in (m.get("seq_a") or ""))
                    fb = " ".join(emo.get(c, "?") for c in (m.get("seq_b") or ""))
                    st.caption(f"Forme Bet261 — {m['home']} : {fa} · ~{m['lam_a']} buts "
                               f"| {m['away']} : {fb} · ~{m['lam_b']} "
                               f"→ **{m['attendus']} buts attendus** au total.")
                    st.markdown("---")
                st.caption("Mesuré sur 59 670 matchs (moitié TRAIN / moitié TEST chrono) : "
                           "le haut de mon classement touche **28%** (pour 27% annoncé) "
                           "contre **23,5%** pour l'ensemble des matchs — le tri apporte "
                           "vraiment. Mais le marché « Total de buts » porte ~11% de marge : "
                           "ROI ≈ −10%, donc mise petite.")

    # ---- 🧭 QUE JOUER ? — conseil tous marchés sur une rencontre choisie ----
    with st.expander("🧭 Que jouer sur ce match ? — mon conseil, tous marchés"):
        import predict_trio as _ptc2
        engC = st.cache_resource(_engine)()
        st.caption("Choisis une rencontre : j'analyse TOUS les marchés (vainqueur, "
                   "total de buts, over/under, les deux marquent, multi-buts, score "
                   "exact, minute du 1er but, 1re équipe à marquer) et je te dis quoi "
                   "jouer. Prédiction issue de la seule forme des équipes.")
        c_lgs = st.multiselect("Ligues (vide = les 9)", list(LEAGUES), default=[], key="cs_lgs")
        cc1, cc2 = st.columns([1, 2])
        c_h = cc1.text_input("Heure (HH:MM Mada) — vide = toutes", value="",
                             key="cs_h", placeholder="ex: 21:03")
        if cc2.button("📥 Charger les rencontres", key="cs_load"):
            hh = c_h.strip()
            if hh and not re.match(r"^\d{1,2}:\d{2}$", hh):
                st.warning("Heure au format HH:MM (ex: 21:03).")
            else:
                with _db("Recherche des rencontres à venir…"):
                    st.session_state["cs_fx"] = _ptc2.rencontres(
                        engC, leagues=[LEAGUES[k] for k in c_lgs] or None,
                        minutes=240, heure=hh or None)
                st.session_state.pop("cs_res", None)
        fx = st.session_state.get("cs_fx")
        if fx is not None:
            if not fx:
                st.info("Aucune rencontre à venir sur ces critères — élargis les ligues, "
                        "vide l'heure, ou attends le prochain round.")
            else:
                choix = st.selectbox(f"Rencontre ({len(fx)} à venir)",
                                     [f["label"] for f in fx], key="cs_sel")
                if st.button("🧭 Que dois-je jouer ?", key="cs_go", type="primary"):
                    r = next((x for x in fx if x["label"] == choix), None)
                    if r:
                        with _db("Analyse de tous les marchés…"):
                            st.session_state["cs_res"] = _ptc2.conseil(engC, r)
        res_c = st.session_state.get("cs_res")
        if res_c is not None:
            if res_c.get("erreur"):
                st.warning(res_c["erreur"])
            else:
                st.markdown(f"### 🧭 `[{res_c['tag']} {res_c['local']}]` "
                            f"{res_c['home']} vs {res_c['away']}")
                s_ = res_c.get("sur")
                if s_:
                    cot = f" · cote **{s_['odds']:g}**" if s_.get("odds") else                           " · _non coté par le book_"
                    st.success(f"**À jouer : « {s_['sel']} »**　_[{s_['marche']}]_"
                               f"{cot} · ma proba **{s_['p']*100:.0f}%**")
                emo = {"V": "🟢", "N": "⚪", "D": "🔴"}
                fa = " ".join(emo.get(x, "?") for x in (res_c.get("seq_a") or ""))
                fb = " ".join(emo.get(x, "?") for x in (res_c.get("seq_b") or ""))
                jr = f"J{res_c['journee']} · " if res_c.get("journee") else ""
                st.caption(f"{jr}**{res_c['attendus']} buts attendus** — "
                           f"{res_c['home']} : {fa} ~{res_c['lam_a']} · "
                           f"{res_c['away']} : {fb} ~{res_c['lam_b']}.")
                st.markdown("**Tous les marchés, du plus sûr au moins sûr :**")
                for l in res_c["lignes"]:
                    cot = f"cote **{l['odds']:g}**" if l.get("odds") else "_non coté_"
                    st.markdown(f"　• _{l['marche']}_ → **{l['sel']}** — "
                                f"**{l['p']*100:.0f}%** · {cot}")
                    alt = " · ".join(f"{t['sel']} {t['p']*100:.0f}%"
                                     + (f" ({t['odds']:g})" if t.get("odds") else "")
                                     for t in l["top3"][1:])
                    if alt:
                        st.caption(f"　　sinon : {alt}")
                st.caption("Probas calibrées marché par marché sur 59 670 matchs "
                           "(moitié TRAIN / moitié TEST chronologique). Mon conseil "
                           "tient : annoncé 79,7% → **touché 80,1%** sur 29 835 matchs "
                           "jamais vus. Mais le book price tout : ROI ≈ **−7%**. "
                           "Le pari le plus sûr n'est pas un pari gagnant — mise petite. "
                           "La règle « proba × cote » a été testée et écartée : elle "
                           "affiche un gain apparent > 1 quatre fois sur cinq sans "
                           "améliorer le ROI.")

    # ---- 💰 QU'EST-CE QUI TOMBE À CETTE COTE ? (relevé historique 1X2) ----
    with st.expander("💰 Qu'est-ce qui tombe à cette cote ? — relevé historique"):
        import predict_trio as _ptrc
        engR = st.cache_resource(_engine)()
        st.caption("Dépose la capture du match, ou saisis les trois cotes : je ressors "
                   "toutes les rencontres passées qui portaient ces mêmes cotes, et ce "
                   "qu'elles ont donné. C'est un relevé de faits, pas une prédiction.")
        # Lecture d'une capture d'ecran : elle PRE-REMPLIT les trois champs, sans
        # jamais lancer la recherche toute seule. Une lecture fausse doit rester
        # visible et corrigeable -- jamais silencieuse.
        img = st.file_uploader("📷 Ou dépose une capture d'écran — un match, ou toute "
                               "la liste d'un round (je lis les cotes dessus)",
                               type=["png", "jpg", "jpeg", "webp"], key="rc_img")
        if img is not None:
            import lire_cotes as _lc
            try:
                lu = _lc.lire_cotes(img.getvalue())
            except Exception as exc:
                lu = {"cotes": [], "message": f"Lecture impossible : {type(exc).__name__}."}
            st.image(img, caption="Capture déposée", width=380)
            lignes_lues = lu.get("lignes") or []
            if lignes_lues:
                st.success(lu["message"])

                def _lib(i, ln):
                    eq = ln.get("equipes") or f"Rencontre {i}"
                    return f"{eq} — " + " / ".join(f"{c:g}" for c in ln["cotes"])
                libs = [_lib(i, ln) for i, ln in enumerate(lignes_lues, 1)]
                # Une capture de round entier porte une dizaine de rencontres :
                # on laisse choisir laquelle plutot que d'en imposer une.
                if len(libs) == 1:
                    st.caption(f"Rencontre lue : **{libs[0]}**")
                    choisie = 0
                else:
                    choisie = libs.index(st.selectbox(
                        f"Quelle rencontre ? ({len(libs)} lues sur la capture)",
                        libs, key="rc_ligne"))
                if st.button("⬇️ Reprendre ces cotes", key="rc_prendre"):
                    for k, v in zip(("rc_1", "rc_x", "rc_2"),
                                    lignes_lues[choisie]["cotes"]):
                        st.session_state[k] = float(v)
                    st.rerun()
            else:
                st.warning(lu.get("message", "Aucune cote lue."))
            # Diagnostic : ce que l'OCR a renvoye AVANT interpretation. Sans cette
            # trace, une lecture fausse sur une vraie capture est impossible a
            # comprendre a distance -- on ne verrait que le resultat, pas la cause.
            # NB : surtout pas un st.expander ici — on est deja dans celui de
            # l'onglet, et Streamlit interdit de les imbriquer (l'app planterait).
            if lu.get("brut") and st.checkbox(
                    "🔍 Voir le détail de lecture (à m'envoyer si les cotes sont fausses)",
                    value=False, key="rc_diag"):
                st.caption(f"{lu.get('boites', 0)} pavés verts repérés · "
                           f"{len(lu['brut'])} ligne(s) de match.")
                for b in lu["brut"]:
                    lus = " | ".join(repr(t) for t in b["ocr"])
                    interp = " / ".join("—" if v is None else f"{v:g}"
                                        for v in b["interprete"])
                    st.markdown(f"　**Ligne {b['ligne']}** — OCR brut : `{lus}`　"
                                f"→ interprété : **{interp}**")
        rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 1])
        r_1 = rc1.number_input("Cote 1", 1.01, 200.0, 2.05, 0.01, key="rc_1", format="%.2f")
        r_x = rc2.number_input("Cote X", 1.01, 200.0, 3.22, 0.01, key="rc_x", format="%.2f")
        r_2 = rc3.number_input("Cote 2", 1.01, 200.0, 3.81, 0.01, key="rc_2", format="%.2f")
        r_tol = rc4.number_input("Tolérance ±", 0.0, 1.0, 0.05, 0.01, key="rc_tol",
                                 help="0.05 = cotes quasi identiques. Élargis si trop peu "
                                      "de rencontres ressortent.")
        r_lgs = st.multiselect("Ligues (vide = les 9)", list(LEAGUES), default=[], key="rc_lgs")
        r_pair = st.checkbox("Limiter à deux équipes précises", value=False, key="rc_pair")
        r_a = r_b = None
        if r_pair:
            _comp = LEAGUES[r_lgs[0]] if len(r_lgs) == 1 else None
            if not _comp:
                st.caption("Choisis UNE seule ligue ci-dessus pour pouvoir nommer les équipes.")
            else:
                _tm = _ptrc.league_teams(engR, _comp)
                if _tm:
                    pa, pb = st.columns(2)
                    r_a = pa.selectbox("Équipe A", _tm, index=0, key="rc_a")
                    r_b = pb.selectbox("Équipe B", _tm, index=min(1, len(_tm) - 1), key="rc_b")
        if st.button("💰 Voir ce qui est tombé", key="rc_go", type="primary"):
            with _db("Recherche des rencontres à cette cote…"):
                st.session_state["rc_res"] = _ptrc.resultats_a_cette_cote(
                    engR, r_1, r_x, r_2, tol=float(r_tol),
                    leagues=[LEAGUES[k] for k in r_lgs] or None,
                    team_a=r_a, team_b=r_b)
        rr = st.session_state.get("rc_res")
        if rr is not None:
            if rr.get("erreur"):
                st.warning(rr["erreur"])
            elif not rr.get("n"):
                st.info("Aucune rencontre passée à ces cotes. Élargis la tolérance, "
                        "ajoute des ligues, ou décoche la limite aux deux équipes.")
            else:
                R = rr["resume"]
                st.success(f"**{rr['n']} rencontres** ont porté les cotes "
                           f"`{rr['cible'][0]:g} / {rr['cible'][1]:g} / {rr['cible'][2]:g}` "
                           f"(± {rr['tol']:g}) — marge du book : {R['marge']}%")
                st.markdown("**Ce qui est sorti :**")
                for i in R["issues"]:
                    flag = "　⚠️ écart notable" if i["notable"] else ""
                    st.markdown(f"　• **{i['sel']}** (cote {i['cote']:g}) → sorti "
                                f"**{i['n']} fois = {i['reel']:.1f}%**  "
                                f"_[{i['ic_bas']:.0f}–{i['ic_haut']:.0f}%]_  ·  "
                                f"le prix dit {i['devigue']:.1f}%{flag}")
                if R["notables"] == 0:
                    st.caption("✅ Aucun écart significatif : à cette cote, le book est juste. "
                               "Les différences visibles tiennent dans le hasard de "
                               f"{rr['n']} rencontres.")
                else:
                    st.warning(f"{R['notables']} écart(s) hors intervalle — à confirmer sur "
                               "un autre échantillon avant d'en tirer quoi que ce soit. "
                               "Sur ce jeu, tous les écarts testés jusqu'ici se sont "
                               "révélés être du bruit.")
                st.markdown(f"**Les buts** : {R['buts_moyen']} en moyenne · "
                            f"plus de 2,5 : **{R['over25']}%** · plus de 3,5 : {R['over35']}% · "
                            f"les deux marquent : {R['btts']}% · 0-0 : {R['zero']}%")
                if R.get("scores"):
                    st.markdown("**Scores les plus fréquents** : " + " · ".join(
                        f"**{x['score']}** {x['pct']:.0f}%" for x in R["scores"]))
                st.markdown("**Les rencontres, de la plus récente à la plus ancienne :**")
                for m in rr["matchs"][:60]:
                    jr = f"`J{m['journee']}` " if m.get("journee") else ""
                    st.markdown(f"　{jr}`{m['date']}` `[{m['tag']}]` {m['home']} "
                                f"**{m['sa']}-{m['sb']}** {m['away']}　→ **{m['issue']}**")
                if rr["n"] > 60:
                    st.caption(f"({rr['n'] - 60} rencontres plus anciennes non affichées.)")
                st.caption(f"Intervalles à 95% corrigés pour les 3 issues testées à la fois "
                           f"(Bonferroni) : sans cette correction, environ une requête sur "
                           f"sept afficherait un faux « écart notable ». Il faut ~"
                           f"{R['n_pour_5pp']} rencontres pour qu'un écart de 5 points "
                           f"soit seulement détectable.")

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

    # Panneau reduit a l'essentiel (demande user) : ligue + heure du round.
    # Les reglages retires sont FIGES sur la valeur qui etait deja proposee par
    # defaut, celle que l'app recommandait :
    #   - seuil de confiance 70 % : c'etait la position initiale du curseur ;
    #   - "haute confiance seulement" desactive : sinon ~90 % des matchs du round
    #     disparaissent (les rounds concentres sont rares, ~10 %) ;
    #   - suivi auto desactive : il relancait un rerun toutes les 45 s.
    # Un seul bouton au lieu de deux : le champ heure fait deja la distinction,
    # vide = prochain round. Et on garde le declenchement au clic — le fit prend
    # 60-90 s au premier appel, le lancer au chargement figerait la page.
    CONF_RECO = 0.70
    want_conf, hi_only = CONF_RECO, False

    lg_name = st.selectbox("Ligue", list(LEAGUES), index=0)
    lg = LEAGUES[lg_name]
    if lg != "InstantLeague-8035":
        st.caption("ℹ️ Ligue en mode MARCHÉ pur (probas dévigées, calibrées) — V2/V5 sont "
                   "entraînés sur l'anglaise.")

    t_str = st.text_input("Heure Mada du round (ex: 21:03) — vide = prochain", value="", key="rt")
    go_h = st.button("🔮 Prédire", type="primary")

    if go_h:
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
        matches_all = res["matches"]
        HI = 0.32
        shown = [m for m in matches_all if (m.get("confidence") or 0) >= HI] if hi_only else matches_all
        if hi_only and not shown:
            st.info("Aucun match assez concentré dans ce round — normal, ils sont rares (~10%).")

        # Vue pronostics seuls (demande user) : QUI GAGNE, tout simplement --
        # une ligne par match, triee de la plus sure a la moins sure. Pas de
        # double chance ni d'over/under. + les grosses cotes value du round.
        # Vue simple (demande user) : Top 3 du round AVEC score exact, puis le
        # vainqueur pronostique des autres matchs, avec proba et cote.
        pronos = []
        for m in shown:
            oh, od, oa = m["cotes"]
            ph, pd_, pa = m["x12"]
            if ph >= pd_ and ph >= pa:
                issue, pi, ci = m.get("team_a") or "1", ph, oh
            elif pa >= pd_:
                issue, pi, ci = m.get("team_b") or "2", pa, oa
            else:
                issue, pi, ci = "Nul", pd_, od
            t1 = m.get("top1_calibre") or (m.get("consensus_top3") or [(None, 0)])[0]
            pronos.append({"pi": pi, "name": m["match"], "issue": issue, "ci": ci,
                           "conf": m.get("confidence") or 0, "t1": t1,
                           "cs": m.get("consensus_top3") or []})
        # Pari suggere (demande user) : pas le favori ecrase a petite cote --
        # le vainqueur predit le PLUS PROBABLE parmi ceux payes a cote >= 2.0
        # (repli 1.8) : cote elevee mais sure.
        cand = ([r for r in pronos if r["issue"] != "Nul" and r["ci"] >= 2.0]
                or [r for r in pronos if r["issue"] != "Nul" and r["ci"] >= 1.8])
        if cand:
            sg = max(cand, key=lambda r: r["pi"])
            sc3 = " · ".join(f"**{s}** ({p*100:.0f}%)" for s, p in sg["cs"][:3])
            st.markdown("### 🎯 Mon pari suggéré du round — cote élevée mais sûre")
            st.success(f"**{sg['name']}** → **{sg['issue']} gagne** — "
                       f"cote **{sg['ci']:g}** · {sg['pi']*100:.0f}%"
                       + (f"  \nTop-3 scores : {sc3}" if sc3 else ""))
        top3 = sorted(pronos, key=lambda r: -r["conf"])[:3]
        top3_names = {r["name"] for r in top3}
        if top3:
            st.markdown("### 🏆 Top 3 du round — avec score exact")
            for i, r in enumerate(top3, 1):
                sc = (f" · score **{r['t1'][0]}** ({r['t1'][1]*100:.0f}%)"
                      if r["t1"] and r["t1"][0] else "")
                st.markdown(f"**{i}. {r['name']}** → **{r['issue']}** ({r['pi']*100:.0f}%) "
                            f"· cote **{r['ci']:g}**{sc}")
        reste = sorted((r for r in pronos if r["name"] not in top3_names),
                       key=lambda r: -r["pi"])
        if reste:
            st.markdown("**Les autres matchs :**")
            for r in reste:
                st.markdown(f"• **{r['name']}** → **{r['issue']}** "
                            f"({r['pi']*100:.0f}%) · cote **{r['ci']:g}**")
        if not pronos:
            st.warning("Aucun match à prédire sur ce round.")

        # Matchs pieges (demande user) : favori fragile, nul menacant, match
        # chaotique ou moteurs en desaccord -- a eviter ; affiche quand il y en a.
        pieges = []
        for m in shown:
            oh, od, oa = m["cotes"]
            ph, pd_, pa = m["x12"]
            inv = 1 / oh + 1 / od + 1 / oa
            if oh <= oa:
                fav, o_fav, p_fav, pm_fav = m.get("team_a") or "1", oh, ph, (1 / oh) / inv
            else:
                fav, o_fav, p_fav, pm_fav = m.get("team_b") or "2", oa, pa, (1 / oa) / inv
            raisons = []
            if o_fav <= 1.7 and p_fav < pm_fav - 0.05:
                raisons.append(f"favori fragile ({pm_fav*100:.0f}% marché vs "
                               f"{p_fav*100:.0f}% moteur)")
            if pd_ >= 0.30 and o_fav <= 2.2:
                raisons.append(f"nul menaçant ({pd_*100:.0f}%)")
            conf = m.get("confidence") or 0
            if 0 < conf < 0.28:
                raisons.append(f"match chaotique (Top-3 {conf*100:.0f}%)")
            if str(m.get("accord", "")).startswith("1/"):
                raisons.append("moteurs en désaccord")
            if raisons:
                pieges.append((m["match"], fav, o_fav, " · ".join(raisons)))
        if pieges:
            st.markdown("**⚠️ Matchs pièges du round — à éviter**")
            for name, fav, o_fav, why in pieges:
                st.markdown(f"• **{name}** (favori {fav} à {o_fav:g}) — {why}")
            st.caption("Piège = le favori paraît sûr mais le moteur voit un risque élevé "
                       "de nul/surprise, ou le match est illisible.")

        # Grosses cotes value (demande user) : SUGGEREES uniquement quand le
        # moteur estime une victoire d'outsider payee au-dessus de sa proba
        # (p x cote >= 1, cote >= 5) -- donc pas a chaque round -- et sans
        # limite de nombre quand il y en a.
        gros = []
        for m in shown:
            oh, od, oa = m["cotes"]
            ph, pd_, pa = m["x12"]
            for team, p, o in ((m.get("team_a") or "1", ph, oh),
                               (m.get("team_b") or "2", pa, oa)):
                if o >= 5.0 and p * o >= 1.0:
                    gros.append((p * o, p, o, team, m["match"]))
        gros.sort(key=lambda r: -r[0])
        if gros:
            st.markdown("**🔦 Grosses cotes value repérées ce round**")
            for v, p, o, team, name in gros:
                st.markdown(f"• **{name}** → **{team} gagne** — cote **{o:g}** · "
                            f"{p*100:.0f}% (value ×{v:.2f})")
            st.caption("Affiché seulement quand la cote paie au-dessus de la proba estimée "
                       "par le moteur (cote ≥5). Spéculatif — RNG calibré, mise petite.")


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
