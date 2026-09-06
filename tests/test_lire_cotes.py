"""Lecture des cotes sur une capture d'écran Bet261.

Ce module PROPOSE des cotes, il ne décide jamais : l'onglet doit les faire
confirmer. Sont verrouillés ici la détection des pavés verts (indépendante de
l'OCR, donc testable partout), le parsing des chiffres lus, et surtout la
DÉGRADATION : sans OCR, sans pavé, sur une image cassée, on prévient au lieu
d'inventer une cote.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lire_cotes as lc  # noqa: E402

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _capture(vals=("2,05", "3,22", "3,81"), larg=1080, haut=200, vert=(76, 175, 80)):
    """Reproduit la ligne de match Bet261 : logos, noms, pavés de cote verts."""
    im = Image.new("RGB", (larg, haut), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((50, 60, 90, 100), fill=(240, 180, 40))
    d.ellipse((50, 118, 90, 158), fill=(40, 60, 160))
    d.text((110, 60), "Wolverhampton", fill=(20, 20, 20))
    d.text((110, 120), "Everton", fill=(20, 20, 20))
    for i, v in enumerate(vals):
        x = 520 + i * 185
        d.rounded_rectangle((x, 62, x + 170, 158), radius=12, fill=vert)
        d.text((x + 70, 100), v, fill=(255, 255, 255))
    return im


def _octets(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


# ---------- détection des pavés ----------

def test_trois_paves_detectes_dans_lordre():
    bx = lc.pave_verts(_capture())
    assert len(bx) == 3, f"3 pavés attendus, {len(bx)} trouvés"
    assert [b[0] for b in bx] == sorted(b[0] for b in bx), "ordre gauche->droite"
    for x0, y0, x1, y1 in bx:
        assert x1 - x0 > 100 and y1 - y0 > 50


def test_deux_lignes_de_match_donnent_six_paves():
    im = _capture()
    d = ImageDraw.Draw(im)
    for i in range(3):                      # une 2e ligne de match plus bas
        x = 520 + i * 185
        d.rounded_rectangle((x, 10, x + 170, 50), radius=8, fill=(76, 175, 80))
    assert len(lc.pave_verts(im)) == 6


def test_le_vert_sombre_du_fond_ne_compte_pas():
    """Un fond verdâtre terne ne doit pas être pris pour un pavé de cote."""
    im = Image.new("RGB", (400, 200), (60, 80, 62))
    assert lc.pave_verts(im) == []


def test_image_sans_vert():
    assert lc.pave_verts(Image.new("RGB", (300, 120), (255, 255, 255))) == []


# ---------- parsing ----------

@pytest.mark.parametrize("txt,attendu", [
    ("2,05", 2.05), ("2.05", 2.05), ("205", 2.05), ("  3,22 ", 3.22),
    ("381", 3.81), ("1,01", 1.01), ("12,50", 12.5),
])
def test_parsing_des_cotes(txt, attendu):
    assert lc._en_cote(txt) == attendu


@pytest.mark.parametrize("txt", ["", "x", None, "0,5", "0", "999,9"])
def test_parsing_refuse_ce_qui_nest_pas_une_cote(txt):
    assert lc._en_cote(txt) is None


# ---------- dégradation : ne jamais inventer ----------

def test_sans_ocr_on_previent_sans_inventer(monkeypatch):
    """Sans pytesseract : on annonce les pavés vus, mais AUCUNE cote."""
    import builtins
    vrai = builtins.__import__

    def faux(nom, *a, **k):
        if nom == "pytesseract":
            raise ImportError("absent")
        return vrai(nom, *a, **k)
    monkeypatch.setattr(builtins, "__import__", faux)
    r = lc.lire_cotes(_octets(_capture()))
    assert r["cotes"] == [], "aucune cote ne doit être inventée sans OCR"
    assert r["boites"] == 3 and r["ocr_dispo"] is False
    assert "main" in r["message"], "l'utilisateur doit être renvoyé à la saisie manuelle"


def test_capture_sans_pave_previent():
    r = lc.lire_cotes(_octets(Image.new("RGB", (300, 120), (255, 255, 255))))
    assert r["cotes"] == [] and r["boites"] == 0
    assert "Recadre" in r["message"] or "main" in r["message"]


def test_donnees_corrompues_ne_plantent_pas():
    r = lc.lire_cotes(b"ceci n'est pas une image")
    assert r["cotes"] == []
    assert "illisible" in r["message"].lower()


def test_octets_vides_ne_plantent_pas():
    assert lc.lire_cotes(b"")["cotes"] == []
