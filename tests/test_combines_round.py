"""Trois combinés proposés par les moteurs sous le tableau du round, et la
correction du 1X2 de V2 dans le trio (04/10).

« Corrige-le, et en bas du tableau donne-moi 3 paris combinés proposés par
les moteurs. »
"""
import math
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402


# --------------------------------------------------------------------------
# LA CORRECTION : LE 1X2 DE V2 EST BIEN LE MELANGE
# --------------------------------------------------------------------------

def test_le_trio_lit_le_1x2_melange_de_v2(monkeypatch):
    """⚠️ Avant le 04/10, `predict_one` lisait `p_h_bl`, une cle que V2 n'a
    jamais rendue : le repli prenait toujours le Poisson pur."""
    monkeypatch.setattr(pt, "predict_match_v2", lambda *a, **k: {
        "lam_h": None, "lam_a": None,
        "p_h_blend": 0.60, "p_d_blend": 0.25, "p_a_blend": 0.15,
        "p_h_pois": 0.40, "p_d_pois": 0.30, "p_a_pois": 0.30})

    def v5_absent(*a, **k):
        raise RuntimeError("pas de V5 dans ce test")
    monkeypatch.setattr(pt, "predict_match_v5", v5_absent)
    m = pt.predict_one(None, None, object(), "A", "B", 1.8, 3.4, 4.5)
    assert m["x12"] == [0.6, 0.25, 0.15]


def test_sans_melange_le_poisson_reste_le_repli(monkeypatch):
    monkeypatch.setattr(pt, "predict_match_v2", lambda *a, **k: {
        "lam_h": None, "p_h_pois": 0.40, "p_d_pois": 0.30, "p_a_pois": 0.30})
    monkeypatch.setattr(pt, "predict_match_v5",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    m = pt.predict_one(None, None, object(), "A", "B", 1.8, 3.4, 4.5)
    assert m["x12"] == [0.4, 0.3, 0.3]


def test_plus_aucune_cle_fantome():
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for fantome in ('"p_h_bl"', '"p_d_bl"', '"p_a_bl"'):
        assert fantome not in code, fantome


# --------------------------------------------------------------------------
# LES TROIS COMBINES
# --------------------------------------------------------------------------

def _m(a, b, legs):
    """legs : {marche: [(sel, p, cote), ...]} -- le format de `market_board`."""
    return {"match": f"{a} v {b}", "team_a": a, "team_b": b, "board": legs}


ROUND = [
    _m("Leeds", "Reds", {"1X2": [("2", 0.80, 1.18)], "Double Chance": [("X2", 0.90, 1.05)]}),
    _m("Fulham", "Palace", {"1X2": [("1", 0.50, 1.90)], "+/-": [("< 3.5", 0.70, 1.33)]}),
    _m("Everton", "Blues", {"Double Chance": [("1X", 0.62, 1.52)]}),
    _m("Brighton", "Pool", {"G/NG": [("Oui", 0.58, 1.62)]}),
    _m("Burnley", "United", {"+/-": [("< 3.5", 0.66, 1.42)]}),
    _m("WestHam", "Spurs", {"1X2": [("1", 0.47, 2.00)]}),
]


def test_trois_combines_au_plus_de_cote_3():
    cs = pt.combines_round(ROUND)
    assert 1 <= len(cs) <= 3
    for c in cs:
        assert c["cote"] >= 3.0
        assert 2 <= len(c["jambes"]) <= 3


def test_la_chance_du_combine_est_le_produit_des_jambes():
    for c in pt.combines_round(ROUND):
        assert math.isclose(c["p"], math.prod(j["p"] for j in c["jambes"]),
                            abs_tol=1e-4)
        assert math.isclose(c["cote"], math.prod(j["o"] for j in c["jambes"]),
                            abs_tol=0.01)


def test_du_plus_probable_au_moins_probable_sur_des_matchs_distincts():
    cs = pt.combines_round(ROUND)
    ps = [c["p"] for c in cs]
    assert ps == sorted(ps, reverse=True)
    ensembles = [frozenset(j["libelle"].split(" → ")[0] for j in c["jambes"])
                 for c in cs]
    assert len(set(ensembles)) == len(ensembles)
    for c in cs:   # une jambe par match
        matchs = [j["libelle"].split(" → ")[0] for j in c["jambes"]]
        assert len(matchs) == len(set(matchs))


def test_les_reglages_sont_ceux_des_combines_surs_du_tracker():
    """Les combines que le suivi reel mesure : on propose les memes."""
    import trio_tracker as tt
    assert tt.FAMILIES["safe"] == (pt.COMBO_MARKETS, 2, 0.45)
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def combines_round("):src.index("def _accord_lisible(")]
    assert "build_combos(list(matches or []), 3.0, 3, top=top," in bloc
    assert "markets=COMBO_MARKETS, min_legs=2, p_min=0.45" in bloc


def test_une_jambe_trop_incertaine_est_ecartee():
    cs = pt.combines_round([_m("A", "B", {"1X2": [("1", 0.30, 3.5)]}),
                            _m("C", "D", {"1X2": [("2", 0.40, 2.5)]})])
    assert cs == []


def test_un_round_vide_ne_propose_rien():
    assert pt.combines_round([]) == [] and pt.combines_round(None) == []


def test_les_jambes_se_lisent_avec_les_equipes():
    assert pt.libelle_jambe("Leeds v Reds", "1X2", "2") == "Leeds – Reds → 1X2 : Reds"
    assert pt.libelle_jambe("Leeds v Reds", "1X2", "X") == "Leeds – Reds → 1X2 : Nul"
    assert (pt.libelle_jambe("Leeds v Reds", "Double Chance", "X2")
            == "Leeds – Reds → Double Chance : X2 (Nul ou Reds)")
    assert pt.libelle_jambe("Leeds v Reds", "+/-", "< 3.5") == "Leeds – Reds → +/- : < 3.5"
