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


def lignes_de_match(img) -> list:
    """Lignes de match d'une capture de ROUND entier.

    Une capture Bet261 complete ne contient pas que des cotes : le bouton
    « S'inscrire », le panier et la banniere du bas sont verts eux aussi. On ne
    retient donc une bande horizontale que si elle porte TROIS paves de meme
    gabarit -- c'est la signature d'une ligne 1X2, et rien d'autre dans l'ecran
    n'y ressemble. Un pave isole (bouton, panier) ou hors-format (banniere) tombe.

    Retour : [{"boites": [(x0,y0,x1,y1) x3], "gauche": (x0,y0,x1,y1)}], de haut
    en bas ; `gauche` est la zone des noms d'equipes, a gauche des cotes.
    """
    np = _np()
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    masque = (g > VERT_MIN) & (g - r > MARGE_VERT) & (g - b > MARGE_VERT)
    if not masque.any():
        return []
    largeur_max = int(a.shape[1] * LARG_MAX_RATIO)
    lignes = []
    for y0, y1 in _bandes(masque, 1, HAUT_MIN):
        boites = [(x0, y0, x1, y1) for x0, x1 in _bandes(masque[y0:y1], 0, LARG_MIN)
                  if x1 - x0 <= largeur_max]
        if len(boites) < 3:
            continue
        # Meme gabarit : on ecarte ce qui n'a pas la taille de la majorite
        # (le panier rond chevauche la derniere ligne, il ne doit pas compter).
        larg = sorted(bx[2] - bx[0] for bx in boites)
        med = larg[len(larg) // 2]
        gab = [bx for bx in boites if abs((bx[2] - bx[0]) - med) <= 0.30 * med]
        if len(gab) < 3:
            continue
        gab = sorted(gab, key=lambda bx: bx[0])[-3:]     # les cotes sont a droite
        ecarts = [gab[i + 1][0] - gab[i][0] for i in range(2)]
        if max(ecarts) > 2.0 * max(min(ecarts), 1):      # doivent etre regulierement espacees
            continue
        lignes.append({"boites": gab, "gauche": (0, y0, max(gab[0][0] - 6, 1), y1)})
    return lignes


def _mots(img_crop) -> str:
    """OCR d'une zone de texte (noms d'equipes). Vide si l'OCR est indisponible."""
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        t = pytesseract.image_to_string(img_crop.convert("L"), config="--psm 6")
    except Exception:
        return ""
    mots = [m.strip() for m in (t or "").splitlines() if m.strip()]
    return " / ".join(mots[:2])


def _prepare(img_crop, seuil: int, marge: int = 12):
    """Isole les chiffres blancs et rend une image nette, noir sur blanc.

    Tesseract lit mal des chiffres colles au bord et de petite taille : on
    agrandit et on ajoute une marge blanche, sans quoi il rate une ligne sur
    deux sur une capture de telephone.
    """
    np = _np()
    from PIL import Image
    a = np.asarray(img_crop.convert("RGB")).astype(int)
    clair = a.min(axis=2) > seuil
    if clair.sum() < 12:                     # quasi rien de blanc : inutile d'essayer
        return None
    net = Image.fromarray(np.where(clair, 0, 255).astype("uint8"), mode="L")
    net = net.resize((max(net.width * 4, 1), max(net.height * 4, 1)), Image.LANCZOS)
    fond = Image.new("L", (net.width + 2 * marge, net.height + 2 * marge), 255)
    fond.paste(net, (marge, marge))
    return fond


# Cascade de reglages : un pave illisible avec l'un passe souvent avec l'autre.
# psm 7 = une ligne de texte, 8 = un mot, 13 = ligne brute sans segmentation.
_ESSAIS = ((150, "7"), (150, "8"), (110, "7"), (110, "13"), (190, "8"))


def _texte(img_crop) -> str:
    """Premier essai concluant de la cascade. Chaine vide si l'OCR est absent."""
    return (_lire_pave(img_crop) or ("", None))[0]


def _lire_pave(img_crop):
    """(texte brut, cote) du premier reglage qui donne une cote plausible.
    Renvoie le dernier texte lu si aucun ne se parse — utile au diagnostic."""
    try:
        import pytesseract
    except Exception:
        return "", None
    dernier = ""
    for seuil, psm in _ESSAIS:
        prep = _prepare(img_crop, seuil)
        if prep is None:
            continue
        try:
            t = pytesseract.image_to_string(
                prep, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.,").strip()
        except Exception:
            return dernier, None
        if t:
            dernier = t
            v = _en_cote(t)
            if v is not None:
                return t, v
    return dernier, None


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
    """Cotes lues sur une capture — une ligne de match ou un round entier.

    Retour : {lignes, cotes, boites, ocr_dispo, message}. `lignes` porte une
    entree par rencontre reperee ({cotes, equipes}) ; `cotes` reprend celles de
    la premiere ligne, pour l'usage simple a un seul match.
    """
    vide = {"lignes": [], "cotes": [], "boites": 0}
    try:
        from PIL import Image
    except Exception:
        return dict(vide, ocr_dispo=False,
                    message="Pillow n'est pas installé : lecture d'image impossible.")
    try:
        img = Image.open(io.BytesIO(donnees))
        img.load()
    except Exception as exc:
        return dict(vide, ocr_dispo=True, message=f"Image illisible ({type(exc).__name__}).")

    lignes_bx = lignes_de_match(img)
    boites = sum(len(l["boites"]) for l in lignes_bx) or len(pave_verts(img))
    if not lignes_bx:
        return dict(vide, boites=boites, ocr_dispo=True,
                    message="Aucune ligne de match repérée (il faut trois cotes alignées). "
                            "Recadre sur la liste des matchs, ou saisis les cotes à la main.")
    try:
        import pytesseract           # noqa: F401
        dispo = True
    except Exception:
        dispo = False
    if not dispo:
        return dict(vide, boites=boites, ocr_dispo=False,
                    message=f"{len(lignes_bx)} ligne(s) de match repérée(s), mais l'OCR "
                            "n'est pas disponible sur ce serveur : saisis les cotes à la main.")

    lignes, incompletes, brut = [], 0, []
    for i, l in enumerate(lignes_bx, 1):
        paires = [_lire_pave(img.crop(bx)) for bx in l["boites"]]
        lus = [t for t, _ in paires]
        vals = [v for _, v in paires]
        # Trace de ce que l'OCR a REELLEMENT renvoye, avant interpretation :
        # sans elle, une lecture fausse est indiagnosticable a distance.
        brut.append({"ligne": i, "ocr": lus, "interprete": vals})
        cotes = [v for v in vals if v is not None]
        if len(cotes) != 3:
            # Ligne dont une cote manque ou reste illisible : sur une capture
            # Bet261 c'est typiquement le bouton panier qui recouvre la 3e cote
            # du dernier match. On la signale plutot que de deviner le chiffre.
            incompletes += 1
            continue
        equipes = ""
        gx0, gy0, gx1, gy1 = l["gauche"]
        if gx1 - gx0 > 40:
            equipes = _mots(img.crop(l["gauche"]))
        lignes.append({"cotes": cotes, "equipes": equipes})
    if not lignes:
        return dict(vide, boites=boites, ocr_dispo=True, brut=brut,
                    message=f"{len(lignes_bx)} ligne(s) repérée(s) mais aucun chiffre "
                            "lisible — agrandis la capture ou saisis les cotes à la main.")
    msg = f"{len(lignes)} rencontre(s) lue(s)."
    if incompletes:
        msg += (f" {incompletes} ligne(s) ignorée(s) : une cote y est masquée "
                "(souvent par le bouton panier) ou illisible.")
    return {"lignes": lignes, "cotes": lignes[0]["cotes"], "boites": boites,
            "incompletes": incompletes, "ocr_dispo": True, "brut": brut,
            "message": msg + " Vérifie les cotes avant de lancer la recherche."}
