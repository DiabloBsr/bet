"""Parsing du planning ttcup.com (table.standard_matches) + fusion/accumulation.

Aucun réseau : on nourrit parse_snapshot avec un planning fabriqué à la main,
fidèle au format réel (versus_url = noms complets, games = manches ; games vide =
match à venir, avec son heure)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import ttcup_tt as tt  # noqa: E402


def _match(order, slug_a, slug_b, time, games):
    return {"order": order, "date": "20.10.2026", "time": time, "time_zone": "CET",
            "versus_url": f"/players/{slug_a}/vs/{slug_b}/", "games": games}


def _tournoi(pk, hall, matches):
    return {"pk": pk, "hall_hint": hall, "period": "20.10.2026",
            "table": {"standard_matches": {"first_day": matches, "second_day": []}}}


# A bat B 2-1 (joué) ; A vs C programmé (games vide) ; B bat C 2-0 (joué).
SNAP = [_tournoi(100, "Czech 7", [
    _match(1, "alpha-a-1990", "beta-b-1991", "07:40", [[11, 5], [9, 11], [11, 7]]),
    _match(2, "alpha-a-1990", "gamma-c-1992", "08:10", []),
    _match(3, "beta-b-1991", "gamma-c-1992", "08:40", [[11, 6], [11, 8]]),
])]


def test_parse_separe_joues_et_a_venir():
    ms = tt.parse_snapshot(SNAP)
    assert len(tt.matchs_joues(ms)) == 2
    av = tt.matchs_a_venir(ms)
    assert len(av) == 1 and av[0]["home"] == "Alpha A" and av[0]["away"] == "Gamma C"


def test_parse_noms_complets_et_score_en_manches():
    ms = tt.parse_snapshot(SNAP)
    ab = next(m for m in ms if {m["home"], m["away"]} == {"Alpha A", "Beta B"})
    assert ab["home"] == "Alpha A" and ab["away"] == "Beta B"
    assert ab["hs"] == 2 and ab["as_"] == 1 and ab["done"] is True   # 2 manches à 1
    assert ab["date"] == "2026-10-20" and ab["hall"] == "Czech 7"


def test_parse_attache_l_heure_mada():
    ms = tt.parse_snapshot(SNAP)
    av = tt.matchs_a_venir(ms)[0]
    # 08:10 à Prague (CEST, UTC+2) = 09:10 à Mada (UTC+3) le 20/10/2026.
    assert av["time"] == "09:10"
    assert av["dt"].startswith("2026-10-20")


def test_fusionner_met_a_jour_le_match_joue():
    ms = tt.parse_snapshot(SNAP)
    av = tt.matchs_a_venir(ms)[0]                        # A vs C, non joué
    joue = dict(av, hs=2, as_=0, done=True)
    fus = tt.fusionner(ms, [joue])
    ac = next(m for m in fus if m["id"] == av["id"])
    assert ac["done"] is True and ac["hs"] == 2
    assert len(fus) == len(ms)


def test_fusionner_dedoublonne_par_id():
    ms = tt.parse_snapshot(SNAP)
    assert len(tt.fusionner(ms, ms)) == len(ms)


def test_iso_convertit_la_date():
    assert tt._iso("20.10.2026") == "2026-10-20"
    assert tt._iso("bizarre") == "bizarre"


def test_slug_nom_retire_l_annee():
    assert tt._slug_nom("sindler-jaroslav-1997") == "Sindler Jaroslav"


def test_sauver_puis_charger_disque(tmp_path):
    p = str(tmp_path / "h.json")
    assert tt.charger_disque(p) == []
    ms = tt.parse_snapshot(SNAP)
    tt.sauver_disque(p, ms)
    assert len(tt.charger_disque(p)) == len(ms)


def test_collecter_once_accumule(tmp_path, monkeypatch):
    p = str(tmp_path / "h.json")
    monkeypatch.setattr(tt, "fetch_hotblock", lambda timeout=25.0: SNAP)
    total1, neufs1 = tt.collecter_once(p)
    assert total1 == 3 and neufs1 == 2                   # 2 joués comptés une fois
    # 2e cycle : A-C désormais joué -> +1 joué, total inchangé
    snap2 = [_tournoi(100, "Czech 7", [
        _match(1, "alpha-a-1990", "beta-b-1991", "07:40", [[11, 5], [9, 11], [11, 7]]),
        _match(2, "alpha-a-1990", "gamma-c-1992", "08:10", [[11, 4], [11, 9]]),
        _match(3, "beta-b-1991", "gamma-c-1992", "08:40", [[11, 6], [11, 8]]),
    ])]
    monkeypatch.setattr(tt, "fetch_hotblock", lambda timeout=25.0: snap2)
    total2, neufs2 = tt.collecter_once(p)
    assert total2 == 3 and neufs2 == 1


def test_pronostics_rattache_heure(monkeypatch):
    ms = tt.parse_snapshot(SNAP)
    hist = tt.matchs_joues(ms)
    fx = tt.matchs_a_venir(ms)
    pr = tt.pronostics(hist, fx)
    assert pr and pr[0].get("time") == "09:10"           # heure propagée au pronostic


def test_marge_points_depuis_games():
    # A gagne 2-1 mais domine largement en points (11-5, 9-11, 11-3) = +12.
    snap = [_tournoi(1, "Czech 7", [
        _match(1, "a-a-1990", "b-b-1991", "07:40", [[11, 5], [9, 11], [11, 3]])])]
    m = tt.matchs_joues(tt.parse_snapshot(snap))[0]
    assert m["hs"] == 2 and m["as_"] == 1                # 2 manches à 1
    assert m["marge"] == (11 + 9 + 11) - (5 + 11 + 3)    # +12 en points


def test_pronostics_liste_bet261_seulement():
    # Historique avec A,B,C,D ; on ne colle que A-B et C-D -> seuls ceux-là.
    hist = ([{"home": "Alpha A", "away": "Beta B", "hs": 3, "as_": 0, "done": True,
              "date": f"2026-10-{d:02d}", "marge": 20} for d in range(1, 9)]
            + [{"home": "Gamma C", "away": "Delta D", "hs": 0, "as_": 3, "done": True,
                "date": f"2026-10-{d:02d}", "marge": -20} for d in range(9, 17)])
    pr, inc = tt.pronostics_liste(hist, "11:10 Alpha, A - Beta, B\nGamma, C - Delta, D")
    assert len(pr) == 2 and inc == []
    assert pr[0]["time"] == "11:10"
