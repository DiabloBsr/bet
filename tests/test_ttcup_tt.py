"""Parsing du snapshot ttcup.com (grille round-robin) + fusion/accumulation.

Aucun réseau : on nourrit parse_snapshot avec une grille fabriquée à la main,
fidèle au format réel (cellule 'a:b' = manches ligne:colonne, '0:0' = à venir,
diagonale vide, symétrie entre (i,j) et (j,i))."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import ttcup_tt as tt  # noqa: E402


def _tournoi(pk, hall, periode, noms, grille):
    """Construit un tournoi au format hotblock à partir d'une grille carrée de
    chaînes ('2:3', '0:0', '' sur la diagonale)."""
    rows = []
    for i, nom in enumerate(noms):
        rows.append({"player": {"order": i + 1, "full_name": nom},
                     "cells": [{"value": v, "is_empty": v == ""} for v in grille[i]]})
    return {"pk": pk, "hall_hint": hall, "period": periode,
            "table": {"tournament_table": rows}, "matches": []}


# 3 joueurs : A bat B 3-1, A-C pas encore joué, B bat C 3-2.
GRILLE = [["", "3:1", "0:0"],
          ["1:3", "", "3:2"],
          ["0:0", "2:3", ""]]
SNAP = [_tournoi(100, "Czech 7", "20.10.2026", ["A", "B", "C"], GRILLE)]


def test_parse_separe_joues_et_a_venir():
    ms = tt.parse_snapshot(SNAP)
    joues = tt.matchs_joues(ms)
    avenir = tt.matchs_a_venir(ms)
    # 3 paires : A-B (joué), A-C (0:0), B-C (joué) -> 2 joués, 1 à venir.
    assert len(joues) == 2
    assert len(avenir) == 1
    assert avenir[0]["home"] == "A" and avenir[0]["away"] == "C"


def test_parse_lit_le_bon_score_et_sens():
    ms = tt.parse_snapshot(SNAP)
    ab = next(m for m in ms if {m["home"], m["away"]} == {"A", "B"})
    assert ab["home"] == "A" and ab["away"] == "B"       # triangle supérieur par ordre
    assert ab["hs"] == 3 and ab["as_"] == 1              # A gagne 3-1
    assert ab["done"] is True
    assert ab["date"] == "2026-10-20" and ab["hall"] == "Czech 7"


def test_chaque_paire_une_seule_fois():
    ms = tt.parse_snapshot(SNAP)
    paires = [frozenset((m["home"], m["away"])) for m in ms]
    assert len(paires) == len(set(paires)) == 3          # A-B, A-C, B-C, sans doublon


def test_fusionner_met_a_jour_le_match_joue():
    ms = tt.parse_snapshot(SNAP)
    av = tt.matchs_a_venir(ms)[0]                         # A vs C, non joué
    # nouveau snapshot où A-C est maintenant joué 3-0
    joue = dict(av, hs=3, as_=0, done=True)
    fus = tt.fusionner(ms, [joue])
    ac = next(m for m in fus if m["id"] == av["id"])
    assert ac["done"] is True and ac["hs"] == 3          # la version jouée remplace
    assert len(fus) == len(ms)                            # pas de doublon créé


def test_fusionner_dedoublonne_par_id():
    ms = tt.parse_snapshot(SNAP)
    fus = tt.fusionner(ms, ms)                            # même snapshot deux fois
    assert len(fus) == len(ms)


def test_iso_convertit_la_date():
    assert tt._iso("20.10.2026") == "2026-10-20"
    assert tt._iso("bizarre") == "bizarre"               # format inconnu : tel quel


def test_sauver_puis_charger_disque(tmp_path):
    p = str(tmp_path / "h.json")
    assert tt.charger_disque(p) == []                    # absent -> vide
    ms = tt.parse_snapshot(SNAP)
    tt.sauver_disque(p, ms)
    assert len(tt.charger_disque(p)) == len(ms)


def test_collecter_once_accumule(tmp_path, monkeypatch):
    p = str(tmp_path / "h.json")
    monkeypatch.setattr(tt, "fetch_hotblock", lambda timeout=25.0: SNAP)
    total1, neufs1 = tt.collecter_once(p)
    assert total1 == 3 and neufs1 == 2                   # 2 matchs joués comptés une fois
    # un 2e cycle avec le match A-C désormais joué : +1 joué, total inchangé (fusion par id)
    grille2 = [["", "3:1", "3:0"], ["1:3", "", "3:2"], ["0:3", "2:3", ""]]
    snap2 = [_tournoi(100, "Czech 7", "20.10.2026", ["A", "B", "C"], grille2)]
    monkeypatch.setattr(tt, "fetch_hotblock", lambda timeout=25.0: snap2)
    total2, neufs2 = tt.collecter_once(p)
    assert total2 == 3 and neufs2 == 1
