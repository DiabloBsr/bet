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


def _fit(lg="InstantLeague-8035"):
    from scraper.config import load_settings
    from sqlalchemy import create_engine
    import predict_trio as pt
    eng = create_engine(load_settings().db_url)
    m5, v2, n = pt.fit(eng, lg)
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


# Les pastilles de forme vivent dans `forme_pastilles`, SANS Streamlit : ce
# n'est que de la construction de chaine, et les garder ici obligeait les tests
# a importer Streamlit — ce qui fait tomber pytest sur cette machine.
from forme_pastilles import pastilles as _pastilles  # noqa: E402
from forme_pastilles import FORME_STYLES as _FORME  # noqa: E402
from html import escape as _esc  # noqa: E402


def _legende_html(st, texte: str) -> None:
    """Une ligne au look de `st.caption`, mais qui laisse les pastilles intactes.

    ⚠️ `st.caption` pose `opacity: 0.6` sur tout son conteneur (mesure le
    04/10) : les pastilles y sortaient ternies, plus aux couleurs de la photo,
    et aucun style pose sur elles ne peut annuler l'opacite d'un parent. Ici,
    seul le TEXTE est attenue, a la meme teinte qu'une legende.
    """
    st.markdown('<span style="font-size:0.875rem;color:color-mix(in srgb, '
                f'currentColor 60%, transparent)">{texte}</span>',
                unsafe_allow_html=True)


def _tableau_vert(st, lignes: list, reco: list, pourcents) -> None:
    """Un tableau dont les cases de MON pronostic sont en vert (04/10).

    Le vert de la photo, celui des pastilles. `reco[i]` liste les colonnes a
    colorer sur la ligne i ; `pourcents` les colonnes affichees en « % ».
    Toutes les lignes d'un coup, sans ascenseur interne, et « Match »
    epinglee : sur telephone le tableau defile en largeur, et les chiffres
    perdaient leur rencontre de vue.
    """
    vert = (f"background-color: {_FORME['V'][0]}; color: #ffffff; "
            "font-weight: 700")

    def _colore(ligne):
        return [vert if c in reco[ligne.name] else "" for c in ligne.index]
    df = pd.DataFrame(lignes)
    fmt = {k: "{:.0f} %" for k in pourcents if k in df}
    if "Cote" in df:
        fmt["Cote"] = "{:g}"
    sty = df.style.apply(_colore, axis=1).format(fmt, na_rep="")
    # « Score exact » et « Sinon » portent l'annotation 1-1 (🔒💀🔻🔺, et le
    # double 🔻🔻/🔺🔺) : « large » debordait sur ordinateur (Olivio, 16/10).
    # « medium » suffit : en portrait sur telephone, les deux emoji tiennent
    # dans la case sans etre coupes, sans elargir a l'exces ailleurs.
    # `width="medium"` est dispo des Streamlit 1.35.
    _cfg = {"Match": st.column_config.TextColumn(pinned=True)}
    for _c in ("Score exact", "Sinon"):
        if _c in df:
            _cfg[_c] = st.column_config.TextColumn(width="medium")
    st.dataframe(sty, hide_index=True, height=35 * (len(lignes) + 1) + 3,
                 column_config=_cfg)


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
            # ⚠️ La forme des deux equipes AVANT chaque rencontre (29/09).
            # Chargee EN UNE FOIS pour tous les face-a-face : une requete par
            # rencontre ferait soixante allers-retours sur une base que le
            # collecteur ecrit en parallele. Et `_safe` parce que cette lecture
            # ne doit pas priver l'utilisateur du face-a-face lui-meme.
            formes = _safe(_pth.formes_avant_h2h, engine, home, away, leagues,
                           h2h[:n_h2h], n) or {}
            st.markdown(
                f"　{_pastilles('VND')} — les **{n} dernières** de chaque "
                f"équipe **avant** la rencontre, de la plus récente à la plus "
                f"ancienne · victoire, nul, défaite.",
                unsafe_allow_html=True)
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
                # Les 5 resultats de CHAQUE equipe avant CETTE rencontre. Rendu
                # a part : les pastilles sont du HTML, la ligne au-dessus est du
                # markdown -- les melanger obligerait a passer TOUT le bloc en
                # `unsafe_allow_html`, y compris les noms d'equipes qui viennent
                # de la base.
                fr = formes.get(m.get("es")) or {}
                if fr.get("a") or fr.get("b"):
                    # Le nom de la marque ORIENTE la lecture : `home`/`away` sont
                    # les equipes CHOISIES, alors que cette rencontre-la a pu se
                    # jouer dans l'autre sens. On nomme donc explicitement.
                    # Les noms viennent de la base et cette ligne passe en
                    # HTML : echappes, ils ne peuvent rien y injecter.
                    st.markdown(
                        f"　{_pastilles(fr.get('a', ''))} _{_esc(home)}_"
                        f"  {_pastilles(fr.get('b', ''))} _{_esc(away)}_",
                        unsafe_allow_html=True)
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



    # ---- 🧭 QUE JOUER ? — conseil tous marchés, TOUTES les rencontres ----
    with st.expander("🧭 Que jouer ? — mon conseil sur toutes les rencontres"):
        import predict_trio as _ptc2
        engC = st.cache_resource(_engine)()
        st.caption("Choisis la ligue et l'heure : j'analyse TOUS les marchés "
                   "(vainqueur, total de buts, over/under, les deux marquent, "
                   "multi-buts, score exact, minute du 1er but, 1re équipe à marquer) "
                   "de CHAQUE rencontre, et je dis quoi jouer. "
                   "Prédiction issue de la seule forme des équipes.")
        c_lgs = st.multiselect("Ligues (vide = les 9)", list(LEAGUES), default=[], key="cs_lgs")
        cc1, cc2 = st.columns([1, 1])
        c_h = cc1.text_input("Heure (HH:MM Mada) — vide = les prochaines heures",
                             value="", key="cs_h", placeholder="ex: 21:03")
        c_max = cc2.number_input("Rencontres max", 1, 30, 12, 1, key="cs_max",
                                 help="Garde-fou : chaque rencontre demande une "
                                      "analyse complète de ses 11 marchés.")
        # Plus aucun filtre de conseil (Olivio, 15/10) : on affiche TOUTES
        # les rencontres analysées, tous conseils, toutes cotes. Seul
        # « Rencontres max » reste, garde-fou de performance.
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
                # Plus de filtre (Olivio, 15/10) : toutes les rencontres
                # analysées sont affichées, tous conseils, toutes cotes.
                gardees = res_l
                n_vraies = sum(1 for r_c in gardees if not r_c.get("erreur"))
                st.success(
                    f"**{n_vraies} rencontre(s) analysée(s)**"
                    + (f" — {total} à venir en tout, coupées à {len(res_l)} par "
                       "le plafond « Rencontres max »." if total > len(res_l)
                       else "."))
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
                    # Les pastilles de la photo, comme sous les face-a-face : un
                    # seul code couleur dans l'app (Olivio, 04/10). La ligne passe
                    # donc en HTML, et les noms venus de la base y sont echappes.
                    fa = _pastilles(res_c.get("seq_a"), 16)
                    fb = _pastilles(res_c.get("seq_b"), 16)
                    jr = f"J{res_c['journee']} · " if res_c.get("journee") else ""
                    _legende_html(st, f"{jr}**{res_c['attendus']} buts attendus** — "
                                  f"{_esc(res_c['home'])} : {fa} ~{res_c['lam_a']} · "
                                  f"{_esc(res_c['away'])} : {fb} ~{res_c['lam_b']}.")
                    # ---- LES DEUX MI-TEMPS : AFFICHAGE RETIRE (Olivio, 16/10) ----
                    # Le moteur calcule toujours res_c["mi_temps"] (le pari
                    # suggere reste le meilleur des 11 marches), mais on
                    # n'affiche plus les pronostics de periode a la demande
                    # d'Olivio.
                    # Detail RESTREINT (Olivio, 15/10) : seulement le score
                    # exact, le 1X2 et le G/NG. Les 11 marches restent calcules
                    # par le moteur (le pari suggere est le meilleur des 11),
                    # seul l'AFFICHAGE du detail est reduit a ces trois-la.
                    MARCHES_DETAIL = ("Score exact", "1X2", "G/NG")
                    for l in res_c["lignes"]:
                        if l["marche"] not in MARCHES_DETAIL:
                            continue
                        cot = f"cote **{l['odds']:g}**" if l.get("odds") else "_non coté_"
                        # ⚠️ LA FIABILITE DU MARCHE, a cote de la proba du match.
                        # Les deux nombres ne disent pas la meme chose : « 14 % »
                        # repond pour CE match, « touche 11,8 % » dit a quel point
                        # ce marche-la est previsible en general. Sans le second,
                        # un score exact a 14 % se lit comme un 1X2 a 14 %, alors
                        # que l'un sort une fois sur huit et l'autre une sur deux.
                        fi = _ptc2.fiabilite_marche(l["marche"])
                        rep = (f" · _ce marché touche **{fi['reel']*100:.0f}%**_"
                               if fi else "")
                        st.markdown(f"　• _{l['marche']}_ → **{l['sel']}** — "
                                    f"**{l['p']*100:.0f}%** · {cot}{rep}")
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


    # ---- 🏇 COURSES — chevaux & lévriers (Palier 1 : lecture des cotes, 19/10) ----
    #   Donnée Bet261/Sporty-Tech en direct (events?eventCategoryIds=8006|8005).
    #   RNG, et les RÉSULTATS ne sont pas exposés : pas d'historique ni d'edge,
    #   juste la proba DÉVIGÉE de victoire par partant. Voir courses_bet261.py.
    with st.expander("🏇 Courses — chevaux (Dashing Derby) & lévriers (Platinum Hounds)"):
        import courses_bet261 as _crs

        def _mada_h(iso):
            """ISO UTC (…Z) → heure Mada (UTC+3) « HH:MM »."""
            try:
                t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
                return t.astimezone(timezone(timedelta(hours=3))).strftime("%H:%M")
            except Exception:
                return "?"

        st.caption("Cotes **en direct** de Bet261. Pour chaque course, la "
                   "probabilité **dévigée** de victoire par partant (la marge du "
                   "book retirée des cotes). ⚠️ Courses RNG : pas d'edge prouvé — "
                   "comme le football virtuel. Une collecte des **arrivées** "
                   "tourne en fond pour permettre, à terme, une analyse "
                   "indépendante des cotes (biais éventuel du RNG).")
        # Compteur de collecte (lit la table `courses` remplie par _courses_loop).
        # Garde : la table peut ne pas exister au tout premier démarrage.
        try:
            from sqlalchemy import text as _text_c
            with _engine().connect() as _cxc:
                _tot_c = _cxc.execute(_text_c("SELECT COUNT(*) FROM courses")).scalar() or 0
                _res_c = _cxc.execute(_text_c(
                    "SELECT COUNT(*) FROM courses WHERE result_json IS NOT NULL")).scalar() or 0
            st.caption(f"🗃️ Collecte en base : **{_tot_c}** course(s) enregistrée(s), "
                       f"dont **{_res_c}** avec arrivée. (Repart de zéro à chaque "
                       f"redéploiement du Space.)")
        except Exception:
            st.caption("🗃️ Collecte : démarrage… (aucune course encore en base).")
        cc1, cc2 = st.columns([2, 1])
        _disc = cc1.radio("Discipline", list(_crs.CATEGORIES), horizontal=True,
                          key="crs_disc")
        _nb = cc2.number_input("Courses à afficher", 1, 10, 4, 1, key="crs_nb")
        if st.button("🏇 Charger les prochaines courses", key="crs_go", type="primary"):
            try:
                st.session_state["crs_res"] = (
                    _disc, _crs.fetch_courses(_crs.CATEGORIES[_disc]), None)
            except Exception as exc:          # réseau/API : on dégrade proprement
                st.session_state["crs_res"] = (_disc, None, f"{type(exc).__name__}: {exc}")
        _cr = st.session_state.get("crs_res")
        if _cr and _cr[0] == _disc:
            _, _races, _err = _cr
            if _races is None:
                st.error(f"Courses indisponibles (API Bet261 injoignable). {_err or ''}")
            elif not _races:
                st.info("Aucune course à venir pour l'instant — réessaie dans un moment.")
            else:
                st.success(f"**{len(_races)} course(s)** à venir · {_disc} "
                           f"{_crs.DISCIPLINE.get(_crs.CATEGORIES[_disc], '')}.")
                for c in _races[:int(_nb)]:
                    st.markdown(f"#### {_esc(str(c['name']))} — {_mada_h(c['start'])} (Mada)")
                    fav = c.get("favori")
                    if fav:
                        st.success(f"Favori : **n° {fav['num']}** — cote "
                                   f"{fav['odds']:g} · proba dévigée "
                                   f"**{fav['p_devig'] * 100:.0f}%**")
                    _df = pd.DataFrame([{"N°": r["num"], "Cote": r["odds"],
                                         "Proba dévigée": r["p_devig"] * 100}
                                        for r in c["runners"]])
                    st.dataframe(
                        _df.style.format({"Cote": "{:g}", "Proba dévigée": "{:.0f} %"}),
                        hide_index=True,
                        height=35 * (len(_df) + 1) + 3)
                    st.markdown("---")

    # ---- 🏆 TOP 3 DU JOUR PAR SPORT (forme + face-à-face, 20/10) ----
    #   Les 3 pronostics les plus CONFIANTS par sport sur les matchs des
    #   prochains jours. Même modèle forme+face-à-face, sans cote.
    with st.expander("🏆 Top 3 du jour — mes 3 meilleurs pronostics par sport"):
        import sport_reel as _sr3
        from datetime import datetime as _dt3, timezone as _tz3, timedelta as _td3

        st.caption("Mes **3 pronostics les plus confiants par sport** pour la "
                   "**date choisie** (forme récente + face-à-face + **blessures**, "
                   "**sans cote**) — avec mon analyse de chaque match. Les absences "
                   "(Out) comptent en NBA/NFL/NHL (ESPN ne publie pas le foot). "
                   "Fiable surtout en **NBA et tennis**.")
        _today3 = (_dt3.now(_tz3.utc) + _td3(hours=3)).date()   # aujourd'hui (Mada)
        _d3 = st.date_input("📅 Date des matchs (ton choix)", value=_today3, key="sr3_date")
        _d3str = _d3.strftime("%Y%m%d")

        @st.cache_data(ttl=10800, show_spinner=False)
        def _sr_jour(path, home_adv, kind, slug, nuls, jour_str):
            hist = _sr3.historique(path, jours=150, kind=kind, slug=slug)
            jour = _dt3.strptime(jour_str, "%Y%m%d").replace(tzinfo=_tz3.utc)
            fx = _sr3.matchs_du_jour(path, jour, kind=kind, slug=slug)
            bl = _sr3.blessures_du_sport(path) if kind == "team" else {}
            return _sr3.pronostics_a_venir(hist, fx, nuls, home_adv, bl, kind=kind,
                                           hockey=(path == "hockey/nhl"))

        if st.button("🏆 Calculer le top 3", key="sr3_go", type="primary"):
            _par_groupe = {g: [] for g in _sr3.GROUPES}
            with _db("Chargement de tous les sports (forme + face-à-face)…"):
                for _lbl, _c in _sr3.SPORTS.items():
                    try:
                        _pr = _sr_jour(_c["path"], _c["home_adv"], _c["kind"],
                                       _c.get("slug"), _c["nuls"], _d3str)
                    except Exception:
                        _pr = []
                    _g = _sr3.groupe_de(_c)
                    for _p in _pr:
                        _par_groupe.setdefault(_g, []).append((_p, _c["kind"] == "tennis", _lbl))
            st.session_state["sr3_res"] = (_d3str, _par_groupe)
        _res3 = st.session_state.get("sr3_res")
        if _res3 is not None:
            _d3prev, _groupes3 = _res3
            def _pick3(p, tennis):
                if tennis:
                    return _esc(p["home"] if p["pick"] == "home" else p["away"])
                return {"home": f"1 — {_esc(p['home'])}", "draw": "Nul",
                        "away": f"2 — {_esc(p['away'])}"}[p["pick"]]
            def _heure3(iso):
                try:
                    t = _dt3.fromisoformat(str(iso).replace("Z", "+00:00"))
                    return t.astimezone(_tz3(_td3(hours=3))).strftime("%d/%m à %Hh%M")
                except Exception:
                    return "heure à confirmer"
            def _champ3(lbl):
                return lbl.split(" ", 1)[1] if " " in lbl else lbl

            @st.cache_data(ttl=10800, show_spinner=False)
            def _corners_ligue(path):
                return _sr3.stats_corners(path)

            # Corners pour les 3 picks FOOTBALL du top (ESPN : 1 requête/match
            # récent → on ne le fait que pour ces 3 matchs affichés).
            _foot_top = sorted(_groupes3.get("⚽ Football", []),
                               key=lambda t: -t[0]["conf"])[:3]
            if _foot_top and not _foot_top[0][0].get("corners"):
                with _db("Corners des matchs foot du top…"):
                    for _p, _tn, _lbl in _foot_top:
                        try:
                            _sc = _corners_ligue(_sr3.SPORTS[_lbl]["path"])
                            _p["corners"] = _sr3.marche_corners(_sc, _p["home"], _p["away"])
                        except Exception:
                            pass

            st.markdown(f"##### Top 3 pour le {_d3prev[6:8]}/{_d3prev[4:6]}/{_d3prev[0:4]}")
            _vu = False
            for _g in _sr3.GROUPES:
                _lst = sorted(_groupes3.get(_g, []), key=lambda t: -t[0]["conf"])[:3]
                st.markdown(f"#### {_g}")
                if not _lst:
                    st.caption("— aucun match ce jour.")
                    continue
                _vu = True
                for _p, _tn, _lbl in _lst:
                    st.markdown(
                        f"**{_pick3(_p, _tn)}** — confiance **{_p['conf'] * 100:.0f}%**  \n"
                        f"<span style='opacity:.75'>{_esc(_p['home'])} – {_esc(_p['away'])} · "
                        f"{_esc(_champ3(_lbl))} · {_heure3(_p.get('date'))} (Mada)</span>",
                        unsafe_allow_html=True)
                    st.markdown("🔎 " + _sr3.analyse_match(_p))
                if _g == "⚽ Football":
                    st.caption("ℹ️ Corners inclus ci-dessus. Le **buteur** n'est "
                               "pas affichable au foot : ESPN ne publie pas les "
                               "buteurs (dispo seulement en NBA/NHL).")
            if not _vu:
                st.info("Aucun match programmé à cette date, tous sports confondus.")

    # ---- ⚽ SPORT RÉEL (Elo, données ESPN) — football réel, 20/10 ----
    #   Sport réel = vrai signal (≠ RNG virtuel). Modèle Elo sur l'historique
    #   ESPN (gratuit, sans clé). Réplicable NFL/NBA/NHL. Voir sport_reel.py.
    with st.expander("🏅 Sport réel — prédiction par Elo (foot · NFL · NBA · NHL · tennis)"):
        import sport_reel as _sr

        st.caption("**Vrai sport, vrai signal** (≠ RNG virtuel) : un modèle **Elo** "
                   "(notes des équipes/joueurs, avantage du terrain au besoin, nul "
                   "au foot) donne des probabilités de victoire calibrées à partir "
                   "de l'historique **ESPN** (gratuit). ⚠️ Honnête : proches des "
                   "bookmakers, mais ça ne **bat pas le marché** de façon rentable "
                   "— c'est pour prédire et comprendre.")
        _lig = st.selectbox("Ligue / sport", list(_sr.SPORTS), key="sr_lig")
        _cfg = _sr.SPORTS[_lig]
        _path, _nuls = _cfg["path"], _cfg["nuls"]

        st.caption("Le pronostic est fondé sur la **forme récente** (6 derniers "
                   "matchs) et le **face-à-face** direct — **aucune cote**. "
                   "J'affiche **6 matchs** de la date choisie, avec l'heure.")
        from datetime import datetime as _dtsr, timezone as _tzsr, timedelta as _tdsr
        _today_sr = (_dtsr.now(_tzsr.utc) + _tdsr(hours=3)).date()
        _date_sr = st.date_input("📅 Date des matchs", value=_today_sr, key="sr_date")
        _dsrstr = _date_sr.strftime("%Y%m%d")

        @st.cache_data(ttl=10800, show_spinner=False)   # 3 h : l'historique bouge peu
        def _sr_charger(path, home_adv, kind, slug, nuls, jour_str):
            hist = _sr.historique(path, jours=220, kind=kind, slug=slug)
            jour = _dtsr.strptime(jour_str, "%Y%m%d").replace(tzinfo=_tzsr.utc)
            fx = _sr.matchs_du_jour(path, jour, kind=kind, slug=slug)[:6]
            bl = _sr.blessures_du_sport(path) if kind == "team" else {}
            pronos = _sr.pronostics_a_venir(hist, fx, nuls, home_adv, bl, kind=kind,
                                            hockey=(path == "hockey/nhl"))
            bt = _sr.taux_reussite(hist, nuls, home_adv)
            return len(hist), pronos, bt

        if st.button("📊 Pronostiquer", key="sr_go", type="primary"):
            with _db("Historique ESPN + prédiction…"):
                try:
                    st.session_state["sr_res"] = (_lig, _sr_charger(
                        _path, _cfg["home_adv"], _cfg["kind"], _cfg.get("slug"),
                        _nuls, _dsrstr))
                except Exception as exc:
                    st.session_state["sr_res"] = (_lig, None)
                    st.session_state["sr_err"] = f"{type(exc).__name__}: {exc}"
        _r = st.session_state.get("sr_res")
        if _r and _r[0] == _lig:
            if _r[1] is None:
                st.error(f"Données ESPN indisponibles. {st.session_state.get('sr_err','')}")
            else:
                _nb, _pronos, _bt = _r[1]
                _tennis = _cfg["kind"] == "tennis"

                def _heure_sr(iso):
                    try:
                        t = _dtsr.fromisoformat(str(iso).replace("Z", "+00:00"))
                        return t.astimezone(_tzsr(_tdsr(hours=3))).strftime("%d/%m %Hh%M")
                    except Exception:
                        return "—"

                def _pick_txt(p):
                    if _tennis:
                        return _esc(p["home"] if p["pick"] == "home" else p["away"])
                    return {"home": f"1 — {_esc(p['home'])}", "draw": "Nul",
                            "away": f"2 — {_esc(p['away'])}"}[p["pick"]]
                st.success(f"**{len(_pronos)}** match(s) le "
                           f"{_date_sr.strftime('%d/%m/%Y')} (6 max).")
                if _bt:
                    st.caption(f"Fiabilité du modèle (backtest) : "
                               f"**{_bt['taux'] * 100:.0f}%** de réussite.")
                if _pronos:
                    _rows = [{"Heure": _heure_sr(p.get("date")),
                              "Match": f"{p['home']} – {p['away']}",
                              "Pronostic": _pick_txt(p),
                              "Confiance": p["conf"] * 100} for p in _pronos]
                    _dfp = pd.DataFrame(_rows)
                    st.dataframe(_dfp.style.format({"Confiance": "{:.0f} %"}),
                                 hide_index=True, height=35 * (len(_dfp) + 1) + 3)
                else:
                    st.info("Aucun match programmé à cette date pour cette ligue.")

    # ---- 🏓 TENNIS DE TABLE (WTT) — forme + face-à-face, 20/10 ----
    #   Source World Table Tennis (résultats officiels, Brotli JSON). Duel entre
    #   deux joueurs. Réutilise le moteur sport_reel. Voir wtt_tt.py.
    with st.expander("🏓 Tennis de table — duel entre deux joueurs (données WTT)"):
        import wtt_tt as _tt
        import sport_reel as _srtt

        st.caption("Résultats officiels **World Table Tennis**, **sans cote** : "
                   "modèle **forme récente + face-à-face** (+ total de manches). "
                   "⚠️ Les tournois archivés disponibles sont surtout des **WTT "
                   "Youth** (catégories jeunes). 1er chargement un peu long, puis "
                   "en cache.")

        @st.cache_data(ttl=10800, show_spinner=False)
        def _tt_charger(annee):
            m = _tt.historique(annees=(annee,), max_events=15)
            bt = _srtt.taux_reussite(m, nuls=False, home_adv=0)
            ratings = dict(_srtt.Elo(home_adv=0).entrainer(m).r)
            return m, bt, ratings

        if st.button("📊 Charger les résultats WTT & entraîner", key="tt_go", type="primary"):
            with _db("Résultats WTT (Brotli) + entraînement…"):
                try:
                    st.session_state["tt_res"] = _tt_charger(2026)
                except Exception as exc:
                    st.session_state["tt_res"] = None
                    st.session_state["tt_err"] = f"{type(exc).__name__}: {exc}"
        _tr = st.session_state.get("tt_res")
        if _tr is not None:
            _m, _bt, _ratings = _tr
            st.success(f"Entraîné sur **{len(_m)}** matchs de simple "
                       f"({len(_ratings)} joueurs).")
            if _bt:
                st.info(f"🎯 **Taux de réussite** du pronostic forme + face-à-face "
                        f"(backtest sur {_bt['tot']} matchs) : **{_bt['taux'] * 100:.0f}%** "
                        f"· référence pile ou face 50%.")
            _js = sorted(_ratings, key=lambda p: -_ratings[p])    # meilleurs d'abord
            if len(_js) >= 2:
                _ca, _cb = st.columns(2)
                _ja = _ca.selectbox("Joueur 1", _js, index=0, key="tt_ja")
                _jb = _cb.selectbox("Joueur 2", _js, index=1, key="tt_jb")
                if _ja and _jb and _ja != _jb:
                    _p = _tt.pronostic(_m, _ja, _jb)
                    _cible = _ja if _p["pick"] == "home" else _jb
                    st.markdown(f"### 👉 Pronostic : **{_esc(_cible)}** "
                                f"— confiance **{_p['conf'] * 100:.0f}%**")
                    st.markdown("🔎 " + _srtt.analyse_match(_p))
                    st.caption(f"Elo : {_esc(_ja)} {round(_ratings.get(_ja, 1500))} · "
                               f"{_esc(_jb)} {round(_ratings.get(_jb, 1500))}")
            st.caption("**Top 10 Elo** : " + " · ".join(
                f"{_esc(t)} {round(_ratings[t])}" for t in _js[:10]))
        elif st.session_state.get("tt_err"):
            st.error(f"WTT indisponible. {st.session_state['tt_err']}")

    # ---- 🔎 HISTORIQUE & FACE-À-FACE (choix manuel, 9 ligues) ----
    with st.expander("🔎 Historique & face-à-face — deux équipes au choix (9 ligues)"):
        import predict_trio as _pth2
        engH = st.cache_resource(_engine)()
        st.caption('Choisis une ligue et deux équipes → face-à-face direct + 5 derniers matchs de chacune (du + récent au + ancien).')
        # Fraîcheur des données (04/10) : si la collecte en ligne s'est arrêtée
        # (Space endormi), on affichait de vieux matchs comme s'ils étaient
        # récents. On annonce la date du dernier résultat, et on alerte au-delà
        # de 2 jours — au rythme d'un round toutes les 2 min, 2 jours sans
        # résultat ne peut signifier qu'une collecte interrompue.
        _dr = _pth2.derniere_date_resultat(engH)
        if _dr:
            _txt, _age = _dr
            if _age > 2:
                st.warning(f"⚠️ Dernier résultat en base : **{_txt}** (il y a "
                           f"{int(_age)} j). La collecte en ligne est "
                           f"interrompue — les face-à-face ci-dessous ne sont "
                           f"pas à jour.")
            else:
                st.caption(f"🟢 Données à jour — dernier résultat : {_txt}.")
        hl1, hl2, hl3 = st.columns([2, 2, 2])
        # ⚠️ `_dfi` etait calcule par l'onglet de prediction, supprime le 27/09 :
        # ce bloc lisait donc un nom qui n'existait plus. Il est desormais calcule
        # ICI, ou il sert. Une valeur par defaut n'a pas a voyager entre deux
        # ecrans independants.
        _lgn = list(LEAGUES)
        _dfi = next((i for i, k in enumerate(_lgn)
                     if LEAGUES[k] == "InstantLeague-8060"), 0)
        h_lg = hl1.selectbox("Ligue", _lgn, index=_dfi, key="h_lg")
        h_comp = LEAGUES[h_lg]
        # ⚠️ DERRIERE `_db` depuis le 27/09. Cette lecture n'a jamais ete gardee,
        # mais d'autres onglets l'etaient : le defaut restait marginal. Avec deux
        # ecrans au total, c'est la moitie de l'app qui affichait une trace Python
        # quand le scraper tenait un verrou -- la classe de bug qui avait mis le
        # Space en boucle de crash.
        with _db("Chargement des équipes…"):
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
                    with _db("Lecture de l'historique…"):
                        _hist_block(st, engH, h_home, h_away, [h_comp], n=5,
                                    show_ou35=h_ou35)
        else:
            st.info("Pas d'équipes trouvées pour cette ligue.")

    # fit PARESSEUX : ne bloque plus le chargement de la page — il ne se lance
    # qu'au premier clic (spinner ~60-90s), puis reste en cache (instantané).
    # Un modèle EN CACHE PAR LIGUE (11/10). max_entries=5 : on garde les 5
    # dernières ligues consultées, les autres se re-fit au besoin — un modèle
    # est léger, mais on borne pour ne pas accumuler 9 engines + modèles.
    cached_fit = st.cache_resource(max_entries=5)(_fit)

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
    # V2/V5 s'entraînent désormais sur CHAQUE ligue (11/10), chacune ayant son
    # historique coté et sa calibration. L'anglaise reste la plus fournie
    # (collectée depuis plus longtemps). Le 1er fit d'une ligue prend quelques
    # dizaines de secondes, puis reste en cache.
    st.caption("🧠 V2 + V5 entraînés sur cette ligue, arbitrés par le marché. "
               "L'anglaise reste la plus fournie en historique.")

    t_str = st.text_input("Heure Mada du round (ex: 21:03) — vide = prochain", value="", key="rt")
    go_h = st.button("🔮 Prédire", type="primary")

    if go_h:
        target = None
        if go_h and t_str.strip():
            d = re.findall(r"\d+", t_str)
            if len(d) >= 2:
                target = f"{int(d[0]) % 24:02d}:{int(d[1]) % 60:02d}"
        try:
            with st.spinner("Fit V5+V2 sur cette ligue (1er appel ~15-90s selon "
                            "sa taille, puis instantané)…"):
                models = cached_fit(lg)
        except Exception as exc:
            st.error(f"Fit impossible : {exc}"); return
        # Garde-fou : une ligue qui vient de redémarrer une saison a peu de
        # résultats -> le modèle est faible, on le dit plutôt que de laisser
        # croire à une analyse solide.
        if models[3] < 500:
            st.warning(f"⚠️ Peu d'historique pour {lg_name} ({models[3]} matchs) "
                       "— l'analyse V2/V5 est peu fiable, le marché domine.")
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
            # Meme regle que le tableau ci-dessous : une seule ecriture.
            _, issue, pi, ci = _pt.issue_round(m)
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
        # ⚠️ LA PREDICTION DU ROUND EN TABLEAU (04/10), avec l'over/under 2,5.
        # Il remplace les deux listes (le Top 3, puis les autres), dans le
        # meme ordre, et garde le score exact de CHAQUE rencontre avec ses
        # deux suivants (27/09) : `tableau_round` est la seule ecriture du
        # format -- deux ecritures divergeraient a la premiere retouche.
        _cb = res.get("cibles") or {}
        tab = _pt.tableau_round(shown, feu=_cb.get("feu"), glace=_cb.get("glace"),
                                eclair=_cb.get("eclair"), derniers=res.get("derniers"),
                                classement=res.get("classement"))
        if tab["lignes"]:
            st.markdown("### 📋 Pronostics du round")
            # Tableau réduit à l'essentiel (Olivio, 16/10) : colonnes 1/X/2,
            # Over/Under 2,5, Classement et Moteurs O/U retirées de l'affichage
            # (le moteur les calcule toujours). Le classement passe sous le
            # tableau, à côté du nom de chaque équipe des matchs marqués.
            _cols_round = ("Top 3", "Match", "Cote",
                           "Score exact", "Sinon", "Pronostic")
            _lignes_aff = [{k: l[k] for k in _cols_round} for l in tab["lignes"]]
            _tableau_vert(st, _lignes_aff, tab["reco"], ())
            # Les 5 derniers matchs (en pastilles) + le classement de chaque
            # équipe des matchs marqués — sous le tableau, car une grille ne
            # peut pas afficher de pastilles HTML. Noms échappés (ligne HTML).
            _formes = res.get("formes") or {}
            _cl = res.get("classement") or {}

            def _nom_rang(t):
                r = _cl.get(t)
                return f"{_esc(t)} ({_pt._rang(r)})" if r else _esc(t)

            if tab.get("marques"):
                st.caption("**Forme des matchs marqués** — 5 derniers, du plus "
                           "récent au plus ancien ; le rang en saison entre "
                           "parenthèses :")
                for _mq in tab["marques"]:
                    _a, _b = _mq["a"], _mq["b"]
                    st.markdown(
                        f"　{_pastilles(_formes.get(_a, ''))} _{_nom_rang(_a)}_"
                        f"  　{_pastilles(_formes.get(_b, ''))} _{_nom_rang(_b)}_",
                        unsafe_allow_html=True)
            st.caption("🔥 = l'équipe en tête de la saison en cours ; 💥 = la "
                       "meilleure attaque ; ⚡ = la meilleure forme récente. "
                       "🏆 = les 3 matchs du round dont le score exact est le "
                       "plus concentré. Le score exact ne sort juste "
                       "qu'environ une fois sur huit.")
            # Légende des annotations 1-1, affichée seulement là où elles
            # s'appliquent (CAN) : inutile de l'imposer aux autres ligues.
            if res.get("derniers"):
                st.caption("Sur un pronostic **1-1**, d'après le dernier match "
                           "de chaque équipe : 🔒 les deux ont fait 0-0 · "
                           "💀 les deux ont perdu sans marquer · "
                           "🔻 domicile 0-0 et extérieur battu (🔻🔻 si 0-1) · "
                           "🔺 extérieur 0-0 et domicile vainqueur (🔺🔺 si 1-0).")
                st.caption("Un pronostic **2-1 ou 1-2** suit les mêmes règles "
                           "(🔒 💀 🔻 🔺, d'après le dernier match de chaque équipe) ; "
                           "si aucune ne s'applique : ⚽.")

        # Le combiné de 3 matchs a été RETIRÉ de l'écran (Olivio, 16/10) ;
        # `combine_round` reste dans le moteur, non câblé.

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

        # Grosses cotes (demande user) : signalees quand le moteur estime une
        # victoire d'outsider payee au-dessus de sa proba (p x cote >= 1,
        # cote >= 5) -- donc pas a chaque round -- et sans limite de nombre.
        #
        # ⚠️ LE MOT « VALUE » A ETE RETIRE LE 27/09, ET RIEN D'AUTRE.
        #
        # La selection, le tri et le nombre affiche sont INCHANGES : c'est bien
        # le bloc d'origine, restaure a l'identique. Seul le libelle bouge.
        #
        # La regle « p x cote >= 1 » a ete testee DEUX fois dans ce depot sur
        # l'historique complet, et s'est revelee un signal INVERSE : elle
        # selectionne les matchs ou le modele s'ecarte le plus du book en sa
        # faveur, c'est-a-dire ses propres erreurs d'estimation. Les paniers
        # qu'elle designait ont fait MOINS bien que le tirage au sort.
        #
        # Continuer a ecrire « value reperee » affirmerait l'inverse de ce qui
        # a ete mesure. Le nombre reste affiche, sous son vrai nom, et la
        # mention dit ce qu'on en sait.
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
            st.markdown("**🔦 Grosses cotes de ce round**")
            for v, p, o, team, name in gros:
                st.markdown(f"• **{name}** → **{team} gagne** — cote **{o:g}** · "
                            f"{p*100:.0f}% (proba × cote = {v:.2f})")
            st.caption("⚠️ Ce n'est PAS une value. La règle qui sélectionne ces "
                       "lignes — proba × cote ≥ 1, cote ≥ 5 — a été testée deux "
                       "fois sur l'historique complet et s'est révélée un signal "
                       "**inversé** : elle retient les matchs où le modèle "
                       "s'écarte le plus du book, donc ses propres erreurs. "
                       "À lire comme un repère de grosses cotes, rien de plus.")

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
