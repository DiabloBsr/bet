"""La forme des deux equipes AVANT chaque face-a-face, en pastilles (29/09).

« Dans l'onglet historique face-a-face, montre comme la couleur sur cette
photo le resultat des 5 dernieres rencontres avant leur face-a-face dans
chaque round. »

Vert a coche = victoire, rouge a croix = defaite, gris a tiret = nul.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt          # noqa: E402
from forme_pastilles import FORME_STYLES, pastilles  # noqa: E402


# --------------------------------------------------------------------------
# LES PASTILLES
# --------------------------------------------------------------------------

def test_trois_couleurs_distinctes():
    v, n, d = (FORME_STYLES[k][0] for k in ("V", "N", "D"))
    assert len({v, n, d}) == 3, "les trois etats doivent se distinguer"


def test_chaque_resultat_rend_une_pastille():
    assert pastilles("VND").count("<span") == 3
    assert pastilles("VNDVV").count("<span") == 5


def test_l_ordre_est_conserve():
    """Du plus RECENT au plus ancien : inverser la suite inverse le rendu."""
    a, b = pastilles("VND"), pastilles("DNV")
    assert a != b
    assert a.index(FORME_STYLES["V"][0]) < a.index(FORME_STYLES["D"][0])
    assert b.index(FORME_STYLES["D"][0]) < b.index(FORME_STYLES["V"][0])


def test_les_glyphes_correspondent_a_la_maquette():
    """Coche = une ligne brisee, croix = deux traits, tiret = un trait
    horizontal. Dessines, car les caracteres sortaient trop fins (04/10)."""
    assert "<polyline" in FORME_STYLES["V"][1]                 # coche
    assert FORME_STYLES["D"][1].count("M") == 2                # croix
    assert FORME_STYLES["N"][1] == '<path d="M7 12H17"/>'      # tiret
    assert [FORME_STYLES[k][2] for k in "VND"] == ["victoire", "nul", "défaite"]


def test_les_couleurs_sont_celles_de_la_photo():
    """Relevees sur la photo d'Olivio (mediane des pixels), pas a l'oeil."""
    assert FORME_STYLES["V"][0] == "#039e52"
    assert FORME_STYLES["D"][0] == "#d96161"
    assert FORME_STYLES["N"][0] == "#bfbfbf"


def test_le_trait_est_blanc_et_epais():
    html = pastilles("V")
    assert 'stroke="#fff"' in html and 'stroke-width="3"' in html


def test_rien_d_autre_que_VND_ne_passe():
    """⚠️ C'est ce qui rend l'insertion HTML sure : seules trois lettres
    connues traversent, jamais une valeur venue de la base."""
    for bruit in ("<script>alert(1)</script>", "VND<img src=x>", "abc", "", None):
        html = pastilles(bruit)
        assert "<script" not in html and "<img" not in html, bruit
        assert html.count("<span") == sum(1 for c in str(bruit or "") if c in "VND")


def test_une_suite_vide_ne_rend_rien():
    assert pastilles("") == "" and pastilles(None) == ""


def test_la_taille_est_reglable():
    assert "width:16px" in pastilles("V", 16)
    assert "width:24px" in pastilles("V", 24)


def test_le_module_n_importe_pas_streamlit():
    """Les tests tombaient sans trace quand ces pastilles vivaient dans le
    tableau de bord : importer Streamlit pour construire une chaine n'a pas
    de sens, et sur cette machine cela fait crasher pytest."""
    import ast
    src = (RACINE / "scripts" / "forme_pastilles.py").read_text(encoding="utf-8")
    # Par l'ARBRE, pas par recherche de chaine : le mot « Streamlit » apparait
    # dans la prose du module, qui explique justement pourquoi il n'en depend
    # pas. Une recherche naive echouait donc sur son propre commentaire.
    importes = {(a.asname or a.name).split(".")[0]
                for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert "streamlit" not in importes, importes
    assert "st" not in importes


# --------------------------------------------------------------------------
# LA FORME D'AVANT CHAQUE FACE-A-FACE
# --------------------------------------------------------------------------

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def base_simulee(monkeypatch):
    """Historique inline : A et B jouent contre des tiers, et se croisent.

    Les dates sont des chaines de meme format que la base (tri lexicographique
    = tri chronologique), ce que `formes_avant_h2h` exploite.
    """
    lignes = [
        # es,                  team_a,   score_a, score_b, team_b
        ("2026-07-04 10:00:00", "A", 2, 0, "T1"),   # A gagne
        ("2026-07-04 10:02:00", "B", 0, 1, "T2"),   # B perd
        ("2026-07-04 10:04:00", "A", 1, 1, "T3"),   # A nul
        ("2026-07-04 10:06:00", "T4", 3, 0, "B"),   # B perd
        ("2026-07-04 10:08:00", "A", 0, 0, "B"),    # FACE-A-FACE n1
        ("2026-07-04 10:10:00", "A", 0, 2, "T5"),   # A perd
        ("2026-07-04 10:12:00", "B", 4, 1, "T6"),   # B gagne
        ("2026-07-04 10:14:00", "B", 1, 0, "A"),    # FACE-A-FACE n2
    ]

    def faux_read_sql(q, engine):
        equipe = "A" if "team_a='A'" in q or "team_b='A'" in q else "B"
        lg = [(es, ta, sa, sb) for es, ta, sa, sb, tb in lignes
              if ta == equipe or tb == equipe]
        return pd.DataFrame(
            [{"es": es, "ta": ta, "sa": sa, "sb": sb} for es, ta, sa, sb in lg]
        ).sort_values("es", ascending=False)
    monkeypatch.setattr(pt.pd, "read_sql", faux_read_sql)
    return lignes


def _h2h(*dates):
    return [{"es": d} for d in dates]


def test_la_forme_s_arrete_avant_la_rencontre(base_simulee):
    """⚠️ Un match ne fait PAS partie de sa propre forme. Sinon on lirait le
    resultat qu'on pretend annoncer."""
    f = pt.formes_avant_h2h(None, "A", "B", None, _h2h("2026-07-04 10:08:00"), n=5)
    d = f["2026-07-04 10:08:00"]
    # A : nul (10:04) puis victoire (10:00) — du plus recent au plus ancien.
    assert d["a"] == "NV"
    # B : defaite (10:06) puis defaite (10:02).
    assert d["b"] == "DD"


def test_chaque_rencontre_a_SA_propre_forme(base_simulee):
    """C'est tout l'interet : la meme paire, lue a deux moments, ne donne pas
    la meme chose."""
    f = pt.formes_avant_h2h(None, "A", "B", None,
                            _h2h("2026-07-04 10:14:00", "2026-07-04 10:08:00"), n=5)
    assert f["2026-07-04 10:08:00"]["a"] == "NV"
    # Avant le second : defaite (10:10), le nul du face-a-face (10:08), puis
    # le nul (10:04) et la victoire (10:00).
    assert f["2026-07-04 10:14:00"]["a"] == "DNNV"


def test_le_face_a_face_precedent_compte_dans_la_forme(base_simulee):
    """Il a bien ete joue : l'exclure serait une autre erreur."""
    f = pt.formes_avant_h2h(None, "A", "B", None,
                            _h2h("2026-07-04 10:14:00"), n=5)
    assert "N" in f["2026-07-04 10:14:00"]["a"][:2]


def test_le_nombre_de_resultats_est_plafonne(base_simulee):
    f = pt.formes_avant_h2h(None, "A", "B", None, _h2h("2026-07-04 10:14:00"), n=2)
    assert len(f["2026-07-04 10:14:00"]["a"]) == 2
    assert len(f["2026-07-04 10:14:00"]["b"]) == 2


def test_sans_historique_la_forme_est_vide(base_simulee):
    f = pt.formes_avant_h2h(None, "A", "B", None, _h2h("2026-07-04 09:00:00"), n=5)
    assert f["2026-07-04 09:00:00"] == {"a": "", "b": ""}


def test_aucun_face_a_face_ne_declenche_aucune_requete(monkeypatch):
    """Zero rencontre = zero aller-retour sur une base que le collecteur ecrit
    en parallele."""
    appels = []
    monkeypatch.setattr(pt.pd, "read_sql",
                        lambda *a, **k: appels.append(1) or pd.DataFrame())
    assert pt.formes_avant_h2h(None, "A", "B", None, [], n=5) == {}
    assert pt.formes_avant_h2h(None, "A", "B", None, None, n=5) == {}
    assert appels == []


def test_deux_requetes_quel_que_soit_le_nombre_de_rencontres(base_simulee,
                                                             monkeypatch):
    """⚠️ L'historique est charge UNE fois par equipe, puis decoupe en
    memoire. Une requete par rencontre ferait soixante allers-retours."""
    compte = []
    vrai = pt.pd.read_sql
    monkeypatch.setattr(pt.pd, "read_sql",
                        lambda q, e: (compte.append(1), vrai(q, e))[1])
    pt.formes_avant_h2h(None, "A", "B", None,
                        _h2h("2026-07-04 10:14:00", "2026-07-04 10:08:00"), n=5)
    assert len(compte) == 2


def test_une_rencontre_sans_date_est_ignoree_sans_lever(base_simulee):
    f = pt.formes_avant_h2h(None, "A", "B", None,
                            [{"es": None}, {"es": "2026-07-04 10:08:00"}], n=5)
    assert list(f) == ["2026-07-04 10:08:00"]


def test_la_date_brute_voyage_avec_chaque_rencontre():
    """Sans elle, la coupure serait impossible : la date AFFICHEE est
    « 05/07 21:03 », sans annee, donc inutilisable pour ordonner."""
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def _match_rows("):src.index("def formes_avant_h2h(")]
    assert '"es": r.expected_start' in bloc
