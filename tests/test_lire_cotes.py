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


# ---------- capture d'un ROUND ENTIER (10 matchs + éléments verts parasites) ----------

MATCHS = [("Fulham", "2,10", "3,62", "3,26"), ("Manchester Red", "3,56", "3,48", "2,03"),
          ("Wolverhampton", "7,09", "4,29", "1,46"), ("Spurs", "1,50", "4,91", "5,24"),
          ("Bournemouth", "1,80", "3,72", "4,26"), ("Leeds", "3,85", "3,20", "2,05"),
          ("Burnley", "3,69", "3,89", "1,88"), ("London Blues", "1,62", "4,04", "5,15"),
          ("Liverpool", "1,39", "4,82", "7,42"), ("West Ham", "2,96", "3,90", "2,40")]


def _round_complet(panier=True, banniere=True, bouton=True):
    """Réplique fidèle d'une capture Bet261 : la liste des matchs d'un round,
    AVEC les éléments verts qui n'ont rien à voir avec des cotes."""
    W, H = 1000, 2076
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 60), fill=(0, 0, 0))                        # barre de statut
    d.rounded_rectangle((505, 88, 765, 175), radius=8, fill=(230, 40, 50))   # bouton rouge
    if bouton:
        d.rounded_rectangle((775, 88, 980, 175), radius=8, fill=(76, 175, 80))  # « S'inscrire »
    y = 230
    for nom, o1, ox, o2 in MATCHS:
        d.text((100, y + 10), nom, fill=(20, 20, 20))
        for i, v in enumerate((o1, ox, o2)):
            x = 483 + i * 168
            d.rounded_rectangle((x, y, x + 152, y + 92), radius=10, fill=(76, 175, 80))
            d.text((x + 60, y + 40), v, fill=(255, 255, 255))
        y += 140
    if panier:      # recouvre la 3e cote du DERNIER match, comme dans l'app
        d.ellipse((823, 1458, 967, 1602), fill=(10, 110, 60))
    if banniere:
        d.rectangle((10, 1660, W - 10, 1760), fill=(20, 120, 45))
    return im


def test_round_entier_toutes_les_lignes_completes():
    """Les 9 lignes entièrement visibles sortent ; la 10e est masquée par le
    panier dans l'app elle-même, donc elle ne doit PAS être devinée."""
    lg = lc.lignes_de_match(_round_complet())
    assert len(lg) == 9, f"9 lignes complètes attendues, {len(lg)} trouvées"
    ys = [l["boites"][0][1] for l in lg]
    assert ys == sorted(ys), "les lignes doivent sortir de haut en bas"
    for l in lg:
        assert len(l["boites"]) == 3
        xs = [bx[0] for bx in l["boites"]]
        assert xs == sorted(xs), "cotes de gauche à droite"


def test_sans_panier_les_dix_lignes_sortent():
    assert len(lc.lignes_de_match(_round_complet(panier=False))) == 10


def test_bouton_sinscrire_nest_pas_pris_pour_un_match():
    """Un pavé vert isolé n'est pas une ligne de cotes : le filtre doit le voir."""
    avec = len(lc.lignes_de_match(_round_complet(panier=False)))
    sans = len(lc.lignes_de_match(_round_complet(panier=False, bouton=False)))
    assert avec == sans == 10, "le bouton vert ne doit rien ajouter ni retirer"


def test_banniere_verte_ignoree():
    avec = len(lc.lignes_de_match(_round_complet(panier=False)))
    sans = len(lc.lignes_de_match(_round_complet(panier=False, banniere=False)))
    assert avec == sans == 10, "la bannière large ne doit pas compter comme une ligne"


def test_zone_des_noms_est_a_gauche_des_cotes():
    for l in lc.lignes_de_match(_round_complet()):
        gx0, gy0, gx1, gy1 = l["gauche"]
        assert gx1 <= l["boites"][0][0], "la zone des noms doit s'arrêter avant la 1re cote"
        assert gx1 - gx0 > 100, "elle doit être assez large pour porter un nom d'équipe"


def test_round_sans_ocr_annonce_les_lignes_sans_inventer(monkeypatch):
    import builtins
    vrai = builtins.__import__

    def faux(nom, *a, **k):
        if nom == "pytesseract":
            raise ImportError("absent")
        return vrai(nom, *a, **k)
    monkeypatch.setattr(builtins, "__import__", faux)
    r = lc.lire_cotes(_octets(_round_complet()))
    assert r["lignes"] == [] and r["cotes"] == []
    assert "9 ligne" in r["message"], f"le compte des lignes doit être annoncé : {r['message']}"


# ---------- le depot de fichier doit rester possible en ligne ----------

def test_upload_actif_sur_le_space():
    """BUG RÉEL : sur Hugging Face l'app tourne derrière un proxy. Avec la
    protection XSRF de Streamlit active, le POST /_stcore/upload_file échoue et
    le fichier déposé est marqué en rouge sans jamais atteindre le code —
    l'onglet paraissait cassé alors que la lecture d'image fonctionnait.
    Ce flag est donc vital : sans lui, le dépôt de capture est mort en ligne."""
    start = (Path(__file__).resolve().parents[1] / "deploy" / "start_cloud.sh")
    src = start.read_text(encoding="utf-8")
    assert "--server.enableXsrfProtection false" in src, \
        "sans ce flag, aucun fichier ne peut être déposé sur le Space"
    assert "--server.maxUploadSize" in src, "taille d'upload à borner explicitement"


def test_ocr_installe_dans_l_image_du_space():
    """Le lecteur de cotes est inerte sans le binaire tesseract ET sans le pont
    python : les deux doivent voyager avec le déploiement."""
    racine = Path(__file__).resolve().parents[1]
    assert "tesseract-ocr" in (racine / "Dockerfile").read_text(encoding="utf-8")
    assert "pytesseract" in (racine / "requirements.txt").read_text(encoding="utf-8")
    deploy = (racine / "deploy" / "deploy_hf.py").read_text(encoding="utf-8")
    for f in ('"Dockerfile"', '"requirements.txt"', '"scripts/lire_cotes.py"'):
        assert f in deploy, f"{f} doit être poussé, sinon le Space garde l'ancienne image"


# ---------- robustesse de lecture : la cascade de reglages ----------

def test_cascade_essaie_plusieurs_reglages():
    """Sur une capture de téléphone, un pavé illisible avec un réglage passe
    souvent avec un autre : 6 lignes lues sur 11 avant la cascade."""
    assert len(lc._ESSAIS) >= 3, "une seule tentative ne suffit pas"
    seuils = {s for s, _ in lc._ESSAIS}
    psms = {p for _, p in lc._ESSAIS}
    assert len(seuils) >= 2, "il faut varier le seuil de binarisation"
    assert len(psms) >= 2, "il faut varier le mode de segmentation"


def test_preparation_agrandit_et_marge():
    """Tesseract lit mal des chiffres collés au bord et petits."""
    im = Image.new("RGB", (40, 20), (76, 175, 80))
    ImageDraw.Draw(im).text((8, 4), "2,05", fill=(255, 255, 255))
    prep = lc._prepare(im, 150)
    assert prep is not None
    assert prep.width > 40 * 3 and prep.height > 20 * 3, "l'image doit être agrandie"
    coins = [prep.getpixel((0, 0)), prep.getpixel((prep.width - 1, prep.height - 1))]
    assert all(c > 200 for c in coins), "il faut une marge blanche autour"


def test_preparation_renonce_sur_un_pave_vide():
    """Pavé sans rien de blanc : inutile de lancer l'OCR dessus."""
    assert lc._prepare(Image.new("RGB", (60, 30), (76, 175, 80)), 150) is None


def test_lire_pave_sans_ocr_ne_plante_pas(monkeypatch):
    import builtins
    vrai = builtins.__import__

    def faux(nom, *a, **k):
        if nom == "pytesseract":
            raise ImportError("absent")
        return vrai(nom, *a, **k)
    monkeypatch.setattr(builtins, "__import__", faux)
    im = Image.new("RGB", (60, 30), (76, 175, 80))
    ImageDraw.Draw(im).text((8, 8), "2,05", fill=(255, 255, 255))
    assert lc._lire_pave(im) == ("", None)


def test_trace_conserve_le_dernier_texte_meme_sans_cote():
    """Si aucun réglage ne donne une cote plausible, on garde quand même ce que
    tesseract a lu : c'est la seule prise pour diagnostiquer à distance."""
    texte, cote = lc._lire_pave(Image.new("RGB", (60, 30), (76, 175, 80)))
    assert cote is None and isinstance(texte, str)
