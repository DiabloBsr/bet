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
    st.dataframe(sty, hide_index=True, height=35 * (len(lignes) + 1) + 3,
                 column_config={"Match": st.column_config.TextColumn(pinned=True)})


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
                    # ---- LES DEUX MI-TEMPS (demande du 27/09) ----
                    # Affichees AVANT le detail des marches cotes, et clairement
                    # separees : ce ne sont pas des paris disponibles sur Bet261,
                    # c'est mon pronostic de periode. Melanger les deux ferait
                    # croire qu'on peut les jouer.
                    mt_ = res_c.get("mi_temps") or {}
                    if mt_:
                        mc1, mc2 = st.columns([1, 1])
                        for col, nom in ((mc1, "1re mi-temps"), (mc2, "2e mi-temps")):
                            d_ = mt_.get(nom) or {}
                            x_ = d_.get("x12") or []
                            if not x_:
                                continue
                            sel_, p_ = max(x_, key=lambda kv: kv[1])
                            # « 1 / X / 2 » de PERIODE : qui marque le plus DANS
                            # cette mi-temps. Un « X » ne veut pas dire match nul.
                            qui_ = {"1": res_c["home"], "2": res_c["away"]}.get(
                                sel_, "aucun des deux ne prend l'avantage")
                            col.markdown(f"　**{nom}** — **{qui_}** "
                                         f"(**{p_*100:.0f}%**)")
                            sc_ = " · ".join(f"**{sc}** {pr*100:.0f}%"
                                             for sc, pr in (d_.get("scores") or [])[:3])
                            col.caption(f"　score : {sc_}　·　"
                                        f"{d_.get('attendus')} buts attendus")
                    # Detail COMPLET pour chaque rencontre : les 11 marches et,
                    # sous chacun, ses deux meilleures alternatives avec leur cote.
                    for l in res_c["lignes"]:
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
                st.caption("Mi-temps : mon pronostic de période, **pas un pari "
                           "Bet261** — le book ne cote ni le 1X2 ni le score "
                           "exact d'une mi-temps. Calibré à part sur 103 714 "
                           "matchs (moitié TRAIN chronologique) et vérifié sur "
                           "103 715 jamais vus : écarts annoncé/touché ramenés "
                           "de +4,8 à **+0,1 point** sur le 1X2 de 2e période.")
                st.caption("Probas calibrées marché par marché sur 59 670 matchs "
                           "(moitié TRAIN / moitié TEST chronologique). Mon conseil "
                           "tient : annoncé 79,7% → **touché 80,1%** sur 29 835 matchs "
                           "jamais vus. Mais le book price tout : ROI ≈ **−7%**. "
                           "Le pari le plus sûr n'est pas un pari gagnant — mise petite.")


    # ---- 🔎 DEBUSQUEUR 1X2 « COTE 2 » SUR LES 5 PROCHAINS ROUNDS ----
    with st.expander("🔎 Débusqueur 1X2 à cote 2 — les 5 prochains rounds, "
                     "toutes ligues"):
        import predict_trio as _pte
        engE = st.cache_resource(_engine)()
        st.caption("Je balaie les 5 prochains rounds de chaque ligue et je sors "
                   "les rencontres dont les TROIS cotes commencent par 2 — "
                   "2,00 à 2,99, peu importe l'après-virgule. Mon pronostic 1X2 "
                   "vient de ma seule analyse de la forme Bet261, jamais de la "
                   "cote.")
        # ⚠️ LE FILTRE SUR LES COTES PASSE AVANT L'ANALYSE. Cinq rounds sur
        # neuf ligues font ~450 rencontres ; les analyser toutes demanderait
        # ~900 requetes de forme pour n'en retenir que 0,58 %. Le tri sur les
        # cotes ne coute rien — il lit trois nombres deja charges. Cote moteur,
        # `debusquer_rounds` filtre donc d'abord et n'analyse que les
        # survivantes. C'est ce qui rend cet ecran immediat.
        eb1, eb2, eb3 = st.columns([1, 1, 2])
        e_ent = eb1.number_input("Les 3 cotes commencent par", 1, 9, 2, 1,
                                 key="eq_ent",
                                 help="2 = les trois cotes entre 2,00 et 2,99. "
                                      "Mesuré : 1 060 relevés sur 184 105, "
                                      "soit 0,58 %, sur 8 ligues.")
        e_nr = eb2.number_input("Rounds à venir", 1, 20, 5, 1, key="eq_nr",
                                help="Par LIGUE : les 5 prochaines heures de "
                                     "chacune, pas les 5 prochaines toutes "
                                     "ligues confondues.")
        e_lgs = eb3.multiselect("Ligues (vide = les 9)", list(LEAGUES),
                                default=[], key="eq_lgs")
        if st.button("🔎 Débusquer", key="eq_go", type="primary"):
            with _db("Balayage des prochains rounds…"):
                st.session_state["eq_res"] = _pte.debusquer_rounds(
                    engE, [LEAGUES[k] for k in e_lgs] or None,
                    n_rounds=int(e_nr), entier=int(e_ent))
        res = st.session_state.get("eq_res")
        if res is not None:
            trouvees = res["trouvees"]
            borne = f"{int(e_ent)},00 à {int(e_ent)},99"
            if not res["examinees"]:
                st.info("Aucune rencontre à venir en base — attends le prochain "
                        "round, ou vérifie que le collecteur tourne.")
            elif not trouvees:
                st.warning(f"**Aucune** des {res['examinees']} rencontre(s) des "
                           f"{res['rounds']} prochains rounds ({res['ligues']} "
                           f"ligue(s)) n'a ses trois cotes entre **{borne}**. "
                           f"C'est rare : 0,58 % des relevés en base.")
            else:
                st.success(f"**{len(trouvees)} rencontre(s)** avec les trois cotes "
                           f"entre **{borne}**, sur {res['examinees']} examinée(s) "
                           f"({res['rounds']} rounds · {res['ligues']} ligue(s)).")
            for m in trouvees:
                lgn = next((n for n, c in LEAGUES.items() if c == m.get("ligue")), "")
                jr = f"J{m['journee']} · " if m.get("journee") else ""
                st.markdown(f"#### `[{lgn} · {jr}{m['local']}]` "
                            f"{m['home']} vs {m['away']}")
                if m.get("erreur"):
                    st.caption(f"⚠️ {m['erreur']}")
                    continue
                tri = " · ".join(f"**{k} = {m['cotes'][k]:g}**"
                                   if m["cotes"].get(k) else f"{k} = ?"
                                   for k in ("1", "X", "2"))
                st.markdown(f"　{tri}")
                cot = (f" · cote **{m['odds']:g}**" if m.get("odds")
                       else " · _non coté_")
                st.success(f"**Mon pronostic : {m['equipe']}** _(« {m['sel']} »)_"
                           f"{cot} · ma proba **{m['p']*100:.0f}%**")
                if m["pieges"]:
                    st.warning("⚠️ **Rencontre piégeuse** — "
                               + " · ".join(m["pieges"]))
                pr = " · ".join(f"{k} **{m['probas'][k]*100:.0f}%**"
                                  for k in ("1", "X", "2"))
                # Memes pastilles que « Que jouer ? » ; noms echappes (HTML).
                fa = _pastilles(m.get("seq_a"), 16)
                fb = _pastilles(m.get("seq_b"), 16)
                _legende_html(st, f"{pr}　—　{m['attendus']} buts attendus　·　"
                           f"{_esc(m['home'])} : {fa} · {_esc(m['away'])} : {fb}")
                st.markdown("---")
            if trouvees:
                st.caption("Pronostic issu de MA seule analyse de la forme Bet261, "
                           "calibré sur 59 670 matchs — ce marché touche **50,4%**. "
                           "Trois cotes voisines veut dire match ouvert, pas match "
                           "rentable : la marge du book reste de **6%**.")

    # ---- 🔎 HISTORIQUE & FACE-À-FACE (choix manuel, 9 ligues) ----
    with st.expander("🔎 Historique & face-à-face — deux équipes au choix (9 ligues)"):
        import predict_trio as _pth2
        engH = st.cache_resource(_engine)()
        st.caption('Choisis une ligue et deux équipes → face-à-face direct + 5 derniers matchs de chacune (du + récent au + ancien).')
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
        tab = _pt.tableau_round(shown)
        if tab["lignes"]:
            st.markdown("### 📋 Pronostics du round")
            _tableau_vert(st, tab["lignes"], tab["reco"],
                          ("1", "X", "2", "Over 2,5", "Under 2,5"))
            st.caption("**En vert, mon pronostic** : l'issue 1X2 choisie et le "
                       "côté le plus probable de l'over/under 2,5. 🏆 = les 3 "
                       "matchs du round dont le score exact est le plus "
                       "concentré. L'over/under 2,5 est la recommandation des "
                       "moteurs — V2, V5 et le marché à poids égaux ; « Moteurs "
                       "O/U » dit combien d'entre eux la portent (hors anglaise, "
                       "le marché est seul). Son taux de réussite n'est pas "
                       "encore mesuré. Le score exact ne sort juste qu'environ "
                       "une fois sur huit.")

        # ---- LE COMBINE DE 3 MATCHS PROPOSE PAR LES MOTEURS (04/10) ----
        # Sous le tableau. `combine_round` : le constructeur de l'app, aux
        # marches et au seuil de jambe des combines « surs » du tracker.
        combi = _pt.combine_round(shown)
        if combi:
            st.markdown("### 🎯 Le combiné de 3 matchs proposé par les moteurs")
            jambes = [f"• {j['libelle']} — cote {j['o']:g} "
                      f"({j['p'] * 100:.0f} %)" for j in combi["jambes"]]
            st.markdown(f"**Cote {combi['cote']:g}** · chance "
                        f"**{combi['p'] * 100:.0f} %**  \n"
                        + "  \n".join(jambes))
            st.caption("Le combiné de 3 matchs le plus probable dont la cote "
                       "atteint 3, une jambe par match. Chance d'une jambe = "
                       "cote du book dévigée ; chance du combiné = leur "
                       "produit. Le book prend sa marge sur chaque jambe : à "
                       "cette cote, le rendement attendu reste négatif.")
        elif tab["lignes"]:
            st.caption("🎯 Pas de combiné de 3 matchs à proposer sur ce round : "
                       "trois jambes assez sûres n'atteignent pas la cote 3.")

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
