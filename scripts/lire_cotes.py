"""Lecture des cotes 1X2 sur une capture d'écran Bet261.

Le pari visuel de Bet261 est très régulier : les cotes sont de gros chiffres
blancs dans des pavés VERTS alignés horizontalement. On exploite exactement ça —
on isole les pavés par leur couleur, puis on ne lit que leur intérieur. Cibler la
couleur évite d'avoir à comprendre le reste de la capture (noms d'équipes, logos,
barre de statut), qui ferait dérailler une lecture plein cadre.

Principe de sûreté : ce module PROPOSE des cotes, il ne décide jamais. L'appelant
doit les faire confirmer avant de s'en servir — une lecture erronée doit rester
visible et corrigeable, jamais silencieuse.
"""
from __future__ import annotations

import io
import re

# Vert des pavés de cote : franchement plus vert que rouge/bleu, et assez lumineux.
VERT_MIN, MARGE_VERT = 110, 35
# Un pavé plausible : ni une icône, ni la moitié de l'écran.
LARG_MIN, HAUT_MIN, LARG_MAX_RATIO = 28, 18, 0.60


def _np():
    import numpy as np
    return np


def _bandes(masque, axe, mini):
    """Groupes d'indices contigus où le masque est présent, le long d'un axe."""
    np = _np()
    presence = masque.any(axis=axe)
    bandes, debut = [], None
    for i, v in enumerate(np.append(presence, False)):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            if i - debut >= mini:
                bandes.append((debut, i))
            debut = None
    return bandes


def pave_verts(img) -> list:
    """Boîtes (x0, y0, x1, y1) des pavés verts, de gauche à droite puis de haut en bas."""
    np = _np()
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    masque = (g > VERT_MIN) & (g - r > MARGE_VERT) & (g - b > MARGE_VERT)
    if not masque.any():
        return []
    largeur_max = int(a.shape[1] * LARG_MAX_RATIO)
    boites = []
    for y0, y1 in _bandes(masque, 1, HAUT_MIN):
        ligne = masque[y0:y1]
        for x0, x1 in _bandes(ligne, 0, LARG_MIN):
            if x1 - x0 <= largeur_max:
                boites.append((x0, y0, x1, y1))
    boites.sort(key=lambda bx: (bx[1], bx[0]))
    return boites


def _texte(img_crop) -> str:
    """OCR d'un pavé, chiffres uniquement. Chaîne vide si l'OCR est indisponible."""
    try:
        import pytesseract
    except Exception:
        return ""
    np = _np()
    a = np.asarray(img_crop.convert("RGB")).astype(int)
    # Le chiffre est BLANC sur le vert : on ne garde que les pixels clairs.
    clair = (a.min(axis=2) > 150)
    if not clair.any():
        return ""
    from PIL import Image
    net = Image.fromarray(np.where(clair, 0, 255).astype("uint8"), mode="L")
    net = net.resize((net.width * 3, net.height * 3), Image.LANCZOS)
    try:
        return pytesseract.image_to_string(
            net, config="--psm 7 -c tessedit_char_whitelist=0123456789.,").strip()
    except Exception:
        return ""


def _en_cote(txt: str):
    """« 2,05 » ou « 2.05 » ou « 205 » -> 2.05. None si ce n'est pas une cote."""
    t = re.sub(r"[^0-9.,]", "", txt or "").replace(",", ".")
    if not t:
        return None
    if "." not in t and len(t) >= 2:          # separateur avale par l'OCR
        t = t[0] + "." + t[1:]
    try:
        v = float(t)
    except ValueError:
        return None
    return round(v, 2) if 1.01 <= v <= 200.0 else None


def lire_cotes(donnees: bytes) -> dict:
    """Cotes lues sur une capture. Retour {cotes, boites, ocr_dispo, message}."""
    try:
        from PIL import Image
    except Exception:
        return {"cotes": [], "boites": 0, "ocr_dispo": False,
                "message": "Pillow n'est pas installé : lecture d'image impossible."}
    try:
        img = Image.open(io.BytesIO(donnees))
        img.load()
    except Exception as exc:
        return {"cotes": [], "boites": 0, "ocr_dispo": True,
                "message": f"Image illisible ({type(exc).__name__})."}
    boites = pave_verts(img)
    if not boites:
        return {"cotes": [], "boites": 0, "ocr_dispo": True,
                "message": "Aucun pavé de cote vert repéré sur cette capture. "
                           "Recadre sur la ligne du match, ou saisis les cotes à la main."}
    try:
        import pytesseract           # noqa: F401
        dispo = True
    except Exception:
        dispo = False
    if not dispo:
        return {"cotes": [], "boites": len(boites), "ocr_dispo": False,
                "message": f"{len(boites)} pavés de cote repérés, mais l'OCR n'est pas "
                           "disponible sur ce serveur : saisis les cotes à la main."}
    cotes = []
    for bx in boites:
        v = _en_cote(_texte(img.crop(bx)))
        if v is not None:
            cotes.append(v)
    if not cotes:
        return {"cotes": [], "boites": len(boites), "ocr_dispo": True,
                "message": f"{len(boites)} pavés repérés mais aucun chiffre lisible — "
                           "agrandis la capture ou saisis les cotes à la main."}
    return {"cotes": cotes, "boites": len(boites), "ocr_dispo": True,
            "message": f"{len(cotes)} cote(s) lue(s) sur {len(boites)} pavé(s). "
                       "Vérifie-les avant de lancer la recherche."}
