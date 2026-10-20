"""Prédiction sport réel par Elo — maths Elo + parsing ESPN, sans réseau."""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import sport_reel as sr  # noqa: E402


# -------- parsing ESPN --------

def _payload(home, away, hs, as_, completed):
    return {"events": [{"date": "2026-01-18T14:00Z", "competitions": [{
        "status": {"type": {"completed": completed}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": home}, "score": str(hs)},
            {"homeAway": "away", "team": {"displayName": away}, "score": str(as_)},
        ]}]}]}


def test_parse_extrait_le_match():
    g = sr.parse_games(_payload("Arsenal", "Everton", 2, 0, True))[0]
    assert g["home"] == "Arsenal" and g["away"] == "Everton"
    assert g["hs"] == 2 and g["as_"] == 0 and g["done"] is True


def test_parse_match_a_venir_sans_score():
    data = {"events": [{"date": "2026-02-01T14:00Z", "competitions": [{
        "status": {"type": {"completed": False}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "A"}, "score": None},
            {"homeAway": "away", "team": {"displayName": "B"}, "score": None}]}]}]}
    g = sr.parse_games(data)[0]
    assert g["done"] is False and g["hs"] is None


# -------- Elo --------

def test_attendu_symetrique_et_borne():
    assert abs(sr.Elo._attendu(1500, 1500) - 0.5) < 1e-9
    assert sr.Elo._attendu(1900, 1500) > 0.9          # bien plus fort -> proba haute
    assert 0 < sr.Elo._attendu(1500, 1900) < 0.1


def test_proba_somme_a_un():
    e = sr.Elo()
    p = e.proba("A", "B", nuls=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert p["draw"] > 0
    p2 = e.proba("A", "B", nuls=False)
    assert p2["draw"] == 0.0 and abs(p2["home"] + p2["away"] - 1.0) < 1e-9


def test_avantage_terrain():
    e = sr.Elo(home_adv=60)
    # deux équipes égales : le domicile est favori grâce à l'avantage terrain.
    p = e.proba("A", "B", nuls=False)
    assert p["home"] > p["away"]


def test_maj_le_vainqueur_monte_le_perdant_descend():
    e = sr.Elo(k=20, home_adv=0)
    e.maj("A", "B", 3, 0)          # A (dom) gagne
    assert e.rating("A") > 1500 and e.rating("B") < 1500
    # somme conservée (k symétrique)
    assert abs((e.rating("A") - 1500) + (e.rating("B") - 1500)) < 1e-9


def test_entrainer_ignore_les_matchs_non_termines():
    games = [
        {"home": "A", "away": "B", "hs": 2, "as_": 0, "done": True},
        {"home": "A", "away": "C", "hs": None, "as_": None, "done": False},  # ignoré
    ]
    e = sr.Elo().entrainer(games)
    assert e.rating("A") > 1500
    assert "C" not in e.r            # le match non joué n'a rien mis à jour


# -------- tennis (1v1, groupings, filtre de tableau) --------

def _tournoi():
    def _m(a, b, a_gagne, done=True):
        return {"id": f"{a}-{b}", "date": "2026-01-18T00:00Z",
                "status": {"type": {"completed": done}},
                "competitors": [{"athlete": {"displayName": a}, "winner": a_gagne},
                                {"athlete": {"displayName": b}, "winner": not a_gagne}]}
    return {"events": [{"date": "2026-01-18T00:00Z", "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [_m("Sinner", "Alcaraz", True)]},
        {"grouping": {"slug": "womens-singles"}, "competitions": [_m("Swiatek", "Gauff", False)]},
        {"grouping": {"slug": "mens-doubles"}, "competitions": [_m("X", "Y", True)]},
    ]}]}


def test_tennis_parse_simple_messieurs_seulement():
    g = sr.parse_games(_tournoi(), kind="tennis", slug="mens-singles")
    assert len(g) == 1
    assert g[0]["home"] == "Sinner" and g[0]["away"] == "Alcaraz"
    assert g[0]["hs"] == 1 and g[0]["as_"] == 0      # Sinner a gagné


def test_tennis_filtre_exclut_doubles_et_autre_genre():
    hommes = sr.parse_games(_tournoi(), kind="tennis", slug="mens-singles")
    femmes = sr.parse_games(_tournoi(), kind="tennis", slug="womens-singles")
    assert [x["home"] for x in hommes] == ["Sinner"]     # pas de doubles, pas de dames
    assert [x["home"] for x in femmes] == ["Swiatek"]
    # sans slug : tous les SIMPLES (hommes + dames), jamais les doubles
    tous = sr.parse_games(_tournoi(), kind="tennis")
    noms = {x["home"] for x in tous}
    assert noms == {"Sinner", "Swiatek"} and "X" not in noms


def test_predire_attache_proba_et_favori():
    e = sr.Elo()
    e.r = {"Fort": 1800, "Faible": 1400}
    out = sr.predire(e, [{"home": "Fort", "away": "Faible", "date": "d", "done": False}])
    assert out[0]["favori"] == "home"
    assert out[0]["p"]["home"] > out[0]["p"]["away"]
    assert out[0]["rating_home"] == 1800


# -------- pronostic FORME + FACE-À-FACE (sans cote) --------

def _g(h, a, hs, as_, d):
    return {"home": h, "away": a, "hs": hs, "as_": as_, "done": True, "date": d}


def test_forme_recent_dabord_et_score():
    passes = [_g("A", "X", 2, 0, "1"), _g("Y", "A", 1, 1, "2"), _g("A", "Z", 0, 3, "3")]
    f = sr.forme(passes, "A", n=3)
    assert f["seq"] == "DNV"            # récent→ancien : défaite, nul, victoire
    assert f["n"] == 3
    assert abs(f["score"] - (0 + 0.5 + 1.0) / 3) < 1e-9


def test_forme_par_lieu_domicile_vs_exterieur():
    # A gagne tous ses matchs À DOMICILE, perd tous ses matchs À L'EXTÉRIEUR.
    passes = ([_g("A", f"h{i}", 2, 0, f"d{i}") for i in range(3)]
              + [_g(f"x{i}", "A", 2, 0, f"e{i}") for i in range(3)])
    assert sr.forme(passes, "A", lieu="home")["score"] == 1.0   # parfait chez lui
    assert sr.forme(passes, "A", lieu="away")["score"] == 0.0   # nul dehors
    assert sr.forme(passes, "A", lieu="home")["n"] == 3
    assert 0.0 < sr.forme(passes, "A")["score"] < 1.0           # mélange sans filtre


def test_pronostic_expose_force_lieu_et_rangs():
    passes = ([_g("A", f"h{i}", 3, 0, f"d{i}") for i in range(4)]
              + [_g(f"x{i}", "B", 3, 0, f"e{i}") for i in range(4)])
    p = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0,
                           rang_home=2, rang_away=9)
    assert p["forme_home_dom"]["n"] == 4 and p["forme_away_ext"]["n"] == 4
    assert p["rang_home"] == 2 and p["rang_away"] == 9


def test_melange_elo_est_la_combinaison_60_40():
    # A domine nettement en forme ; l'Elo, qui pondère par la force des
    # adversaires, est moins tranché → le mélange 60/40 doit tomber ENTRE les
    # deux, exactement à 0,6·Elo + 0,4·forme (pas de nul : home+away=1).
    passes = ([_g("A", "C", 3, 0, f"ac{i}") for i in range(4)]
              + [_g("C", "B", 3, 0, f"cb{i}") for i in range(4)]
              + [_g("A", "D", 1, 1, f"ad{i}") for i in range(2)]
              + [_g("B", "D", 1, 1, f"bd{i}") for i in range(2)])
    sans = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0)
    elo = sr.Elo(home_adv=0).entrainer(passes)
    pe = elo.proba("A", "B", nuls=False)
    avec = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0, elo=elo)
    attendu = 0.6 * pe["home"] + 0.4 * sans["p"]["home"]
    assert abs(avec["p"]["home"] - attendu) < 1e-9
    assert pe["home"] < avec["p"]["home"] < sans["p"]["home"]   # tiré vers l'Elo
    assert abs(sum(avec["p"].values()) - 1.0) < 1e-9            # proba normalisée


def test_classement_lit_l_ordre_des_entrees(monkeypatch):
    faux = {"children": [{"standings": {"entries": [
        {"team": {"displayName": "Alpha"}},
        {"team": {"displayName": "Beta"}},
        {"team": {"displayName": "Gamma"}}]}}]}
    monkeypatch.setattr(sr, "_get", lambda u, t=20.0: faux)
    cl = sr.classement_du_sport("soccer/eng.1")
    assert cl == {"Alpha": 1, "Beta": 2, "Gamma": 3}


def test_face_a_face_compte_du_point_de_vue_de_a():
    passes = [_g("A", "B", 2, 0, "1"), _g("B", "A", 1, 1, "2"), _g("B", "A", 3, 0, "3"),
              _g("A", "C", 1, 0, "4")]   # le dernier (vs C) ne compte pas
    h = sr.face_a_face(passes, "A", "B", n=6)
    assert (h["va"], h["nul"], h["vb"], h["tot"]) == (1, 1, 1, 3)
    assert abs(h["score_a"] - (1 + 0.5) / 3) < 1e-9


def test_pronostic_meilleure_forme_est_favori():
    # A gagne tout récemment, B perd tout -> pronostic = A (home).
    passes = ([_g("A", f"o{i}", 2, 0, f"a{i}") for i in range(6)]
              + [_g("B", f"p{i}", 0, 2, f"b{i}") for i in range(6)])
    pr = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0)
    assert pr["pick"] == "home" and pr["p"]["home"] > 0.5


def test_backtest_rend_un_taux_borne_ou_none():
    assert sr.taux_reussite([], nuls=False) is None      # trop peu
    base = [_g("A", "B", 2, 0, f"{i:03d}") for i in range(80)]
    bt = sr.taux_reussite(base, nuls=False, home_adv=0, warmup=20)
    assert bt is not None and 0.0 <= bt["taux"] <= 1.0 and bt["tot"] > 0


# -------- marchés : totaux, les deux marquent, meilleur marqueur --------

def _gl(h, a, hs, as_, d, lh=None, la=None):
    g = _g(h, a, hs, as_, d)
    g["lead_home"], g["lead_away"] = lh, la
    return g


def test_stats_off_moyennes():
    passes = [_g("A", "X", 2, 1, "1"), _g("Y", "A", 0, 3, "2")]  # A marque 2 puis 3, encaisse 1 puis 0
    bf, bc = sr.stats_off(passes, "A", n=5)
    assert abs(bf - 2.5) < 1e-9 and abs(bc - 0.5) < 1e-9


def test_marches_football_btts_et_over():
    # beaucoup de buts des deux côtés -> BTTS probable, over 2,5 probable.
    passes = ([_g("A", f"o{i}", 2, 1, f"a{i}") for i in range(8)]
              + [_g("B", f"p{i}", 2, 1, f"b{i}") for i in range(8)])
    m = sr.marches_buts(passes, "A", "B", nuls=True, ligue_avg=3.0)
    assert m["ligne"] == 2.5 and "btts" in m
    assert m["over"] > 0.5 and m["btts"] > 0.5
    assert m["exp_total"] > 2.5


def test_marches_nba_total_vs_ligne():
    passes = [_g("A", "B", 120, 110, f"{i:03d}") for i in range(10)]  # total 230 > ligue_avg
    m = sr.marches_buts(passes, "A", "B", nuls=False, ligue_avg=220.0)
    assert "btts" not in m
    assert m["over"] > 0.5                      # équipes qui scorent au-dessus de la ligne


def test_mlb_est_dans_sports_et_groupes():
    assert "⚾ MLB (baseball)" in sr.SPORTS
    cfg = sr.SPORTS["⚾ MLB (baseball)"]
    assert cfg["path"] == "baseball/mlb" and cfg["nuls"] is False and cfg["kind"] == "team"
    assert "⚾ MLB" in sr.GROUPES
    assert sr.groupe_de(cfg) == "⚾ MLB"


def test_unite_du_sport():
    assert sr.unite_du_sport("hockey/nhl") == "buts"
    assert sr.unite_du_sport("baseball/mlb") == "runs"
    assert sr.unite_du_sport("basketball/nba") == "points"


def test_total_mlb_libelle_en_runs():
    hist = [_g("A", "B", 5, 4, f"{i:03d}") for i in range(8)]
    fx = [{"home": "A", "away": "B", "date": "2026-10-21T19:00Z", "done": False}]
    out = sr.pronostics_a_venir(hist, fx, nuls=False, home_adv=24,
                                unite=sr.unite_du_sport("baseball/mlb"))
    assert out[0]["marches"]["unite"] == "runs"
    assert "Total de runs" in sr.analyse_match(out[0])


def test_meilleur_marqueur_choisit_le_meneur_recurrent():
    L = {"nom": "Star", "val": 30.0}
    passes = [_gl("A", "B", 1, 0, f"{i:02d}", lh=L, la={"nom": "Autre", "val": 10.0})
              for i in range(5)]
    mq = sr.meilleur_marqueur(passes, "A", "B", n=10)
    assert mq["home"]["nom"] == "Star" and mq["home"]["fois"] == 5
    assert mq["top"]["nom"] == "Star"           # 30 > 10
    assert mq["top"]["ligne"] == 29.5           # plus de (30 arrondi − 0,5)


def test_meilleur_marqueur_en_buts_lit_le_bon_champ():
    # NHL : on veut le meneur BUTS, pas points.
    passes = [{"home": "A", "away": "B", "hs": 3, "as_": 1, "done": True, "date": f"{i}",
               "lead_home": {"nom": "Passeur", "val": 20.0},
               "but_home": {"nom": "Buteur", "val": 2.0}} for i in range(5)]
    mq = sr.meilleur_marqueur(passes, "A", "B", buts=True)
    assert mq["home"]["nom"] == "Buteur" and mq["buts"] is True
    assert mq["home"]["ligne"] == 1.5           # ~2 buts → plus de 1,5


def test_parse_team_capte_le_meilleur_marqueur():
    data = {"events": [{"date": "d", "competitions": [{
        "status": {"type": {"completed": True}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "A"}, "score": "100",
             "leaders": [{"name": "points", "leaders": [
                 {"athlete": {"displayName": "Joueur X"}, "value": 28}]}]},
            {"homeAway": "away", "team": {"displayName": "B"}, "score": "98",
             "leaders": []}]}]}]}
    g = sr.parse_games(data)[0]
    assert g["lead_home"] == {"nom": "Joueur X", "val": 28.0}
    assert g["lead_away"] is None


# -------- tennis : total de sets --------

def test_parse_tennis_capte_le_nombre_de_sets():
    data = {"events": [{"date": "d", "groupings": [{"grouping": {"slug": "mens-singles"},
        "competitions": [{"status": {"type": {"completed": True}},
            "format": {"regulation": {"periods": 3}},
            "competitors": [
                {"athlete": {"displayName": "A"}, "winner": True,
                 "linescores": [{"value": 6}, {"value": 7}]},
                {"athlete": {"displayName": "B"}, "winner": False,
                 "linescores": [{"value": 4}, {"value": 5}]}]}]}]}]}
    g = sr.parse_games(data, kind="tennis", slug="mens-singles")[0]
    assert g["n_sets"] == 2 and g["periods"] == 3
    # jeux = (6+7) + (4+5) = 22
    assert g["n_games"] == 22


def test_marche_jeux_ligne_selon_best_of():
    passes = [{"home": "A", "away": "C", "n_games": 24, "done": True, "hs": 1, "as_": 0},
              {"home": "B", "away": "D", "n_games": 20, "done": True, "hs": 1, "as_": 0}]
    bo3 = sr.marche_jeux(passes, "A", "B", periods=3)
    assert bo3["ligne"] == 22.5 and abs(bo3["exp"] - 22.0) < 1e-9
    assert sr.marche_jeux(passes, "A", "B", periods=5)["ligne"] == 36.5
    assert sr.marche_jeux([], "A", "B") is None


def test_marche_sets_ligne_selon_best_of():
    passes = [{"home": "A", "away": "C", "n_sets": 3, "done": True, "hs": 1, "as_": 0},
              {"home": "B", "away": "D", "n_sets": 3, "done": True, "hs": 1, "as_": 0}]
    bo3 = sr.marche_sets(passes, "A", "B", periods=3)
    assert bo3["ligne"] == 2.5 and bo3["exp"] == 3.0 and bo3["over"] > 0.5
    bo5 = sr.marche_sets(passes, "A", "B", periods=5)
    assert bo5["ligne"] == 3.5
    assert sr.marche_sets([], "A", "B") is None


# -------- corners (football) --------

def test_corners_match_parse(monkeypatch):
    summary = {"boxscore": {"teams": [
        {"team": {"displayName": "A"}, "statistics": [
            {"name": "wonCorners", "displayValue": "7"}, {"name": "offsides", "displayValue": "2"}]},
        {"team": {"displayName": "B"}, "statistics": [
            {"name": "wonCorners", "displayValue": "4"}]}]}}
    monkeypatch.setattr(sr, "_get", lambda u, t=20.0: summary)
    c = sr._corners_match("soccer/eng.1", 1, 20.0)
    assert c == {"A": 7.0, "B": 4.0}


def test_marche_corners_total_et_ligne():
    sc = {"A": {"for": 7.0, "against": 5.0, "n": 5},
          "B": {"for": 6.0, "against": 4.0, "n": 5}}
    m = sr.marche_corners(sc, "A", "B", ligne=9.5)
    # (7+4)/2 + (6+5)/2 = 5.5 + 5.5 = 11 > 9.5 -> over probable
    assert abs(m["exp"] - 11.0) < 1e-9 and m["ligne"] == 9.5 and m["over"] > 0.5
    assert sr.marche_corners({}, "A", "B") is None     # pas de données


# -------- blessures --------

def test_poids_blessures():
    assert sr.poids_blessures(None) == 0.0
    assert sr.poids_blessures([{"status": "Out"}, {"status": "Out"}]) == 2.0
    assert sr.poids_blessures([{"status": "Day-To-Day"}]) == 0.4
    assert sr.poids_blessures([{"status": "Out"}, {"status": "Day-To-Day"}]) == 1.4


def test_pronostics_portent_la_date():
    hist = [_g("A", "B", 2, 0, f"{i:03d}") for i in range(8)]
    fx = [{"home": "A", "away": "B", "date": "2026-10-21T19:00Z", "done": False}]
    out = sr.pronostics_a_venir(hist, fx, nuls=False, home_adv=0)
    assert out[0]["date"] == "2026-10-21T19:00Z"


def test_nhl_total_en_buts_over_under():
    # NHL (hockey=True) : le total est libellé en « buts », pas en « points ».
    hist = [_g("A", "B", 4, 3, f"{i:03d}") for i in range(8)]
    fx = [{"home": "A", "away": "B", "date": "2026-10-21T19:00Z", "done": False}]
    out = sr.pronostics_a_venir(hist, fx, nuls=False, home_adv=0, hockey=True)
    assert out[0]["marches"]["unite"] == "buts"
    txt = sr.analyse_match(out[0])
    assert "Total de buts" in txt


def test_analyse_cite_le_pronostic_et_la_forme():
    passes = ([_g("A", f"o{i}", 2, 0, f"a{i}") for i in range(6)]
              + [_g("B", f"p{i}", 0, 2, f"b{i}") for i in range(6)])
    p = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0,
                           inj_away=[{"nom": "Star", "status": "Out"}])
    txt = sr.analyse_match(p)
    assert "Mon pronostic" in txt and "A" in txt
    assert "Forme récente" in txt
    assert "Star" in txt            # l'absent est cité


def test_matchs_du_jour_filtre_non_termines(monkeypatch):
    from datetime import datetime, timezone
    jeux = [{"home": "A", "away": "B", "date": "x", "done": False},
            {"home": "C", "away": "D", "date": "y", "done": True},
            {"home": "E", "away": "", "date": "z", "done": False}]
    monkeypatch.setattr(sr, "fetch_range", lambda *a, **k: jeux)
    out = sr.matchs_du_jour("p", datetime(2026, 10, 21, tzinfo=timezone.utc))
    assert [g["home"] for g in out] == ["A"]   # pas le terminé, pas le sans adversaire


def test_blessures_baissent_la_proba_de_l_equipe_touchee():
    # deux camps à forme égale ; le domicile perd 3 titulaires -> sa proba baisse.
    passes = ([_g("A", f"o{i}", 1, 0, f"a{i}") for i in range(6)]
              + [_g("B", f"p{i}", 1, 0, f"b{i}") for i in range(6)])
    sans = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0)
    avec = sr.pronostic_forme(passes, "A", "B", nuls=False, home_adv=0,
                              inj_home=[{"status": "Out"}] * 3)
    assert avec["p"]["home"] < sans["p"]["home"]
    assert len(avec["bless_home"]) == 3
