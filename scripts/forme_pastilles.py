"""Pastilles de forme : une suite de V/N/D en ronds colores.

Module SANS dependance Streamlit, et c'est deliberе : ces pastilles sont de la
construction de chaine, rien d'autre. Les laisser dans le tableau de bord
obligeait les tests a importer Streamlit — ce qui, sur cette machine, fait
tomber pytest sans meme une trace.

── LA MAQUETTE (29/09) ──────────────────────────────────────────────────────

Rond vert a coche pour une victoire, rond rouge a croix pour une defaite, rond
gris a tiret pour un nul.

⚠️ EN HTML ET NON EN EMOJI, et c'est le seul moyen d'etre fidele : aucun emoji
ne porte a la fois la COULEUR et le GLYPHE demandes. La coche verte existe en
carre, la croix rouge n'a pas de rond, et il n'y a pas de rond gris a tiret.
Trois symboles de familles differentes se liraient comme trois choses sans
rapport, alors que la maquette montre une serie.
"""
from __future__ import annotations

# Couleurs RELEVEES sur la photo d'Olivio (mediane des pixels, 04/10), pas
# choisies a l'oeil : la premiere version (#21b35a, #e8384f, #c2c8d0) donnait
# un vert plus vif et un rouge plus dur que la maquette.
#
# ⚠️ Glyphes DESSINES (SVG), plus en caracteres : a 16 px, « ✓ ✕ − » sortaient
# en trait fin de police symbole, que le gras n'epaissit pas. La photo montre
# un trait blanc epais, environ 1/8 du diametre : c'est le `stroke-width` 3
# sur 24 ci-dessous.
FORME_STYLES = {
    # lettre : (couleur, trace dans un carre 24x24, nom lu au survol)
    "V": ("#039e52", '<polyline points="6.5,12.5 10.5,16.5 17.5,8.5"/>', "victoire"),
    "D": ("#d96161", '<path d="M8 8L16 16M16 8L8 16"/>', "défaite"),
    "N": ("#bfbfbf", '<path d="M7 12H17"/>', "nul"),
}


def pastilles(seq, taille: int = 20) -> str:
    """Une suite « VNDVV » -> pastilles rondes, du plus RECENT au plus ancien.

    Seuls V, N et D sont rendus ; tout autre caractere est ignore. C'est aussi
    ce qui rend l'insertion HTML sure : rien venu de la base ne traverse,
    seulement trois lettres connues et des traces fixes.
    """
    t = int(taille)
    bouts = []
    for c in str(seq or ""):
        st = FORME_STYLES.get(c)
        if not st:
            continue
        couleur, trace, nom = st
        bouts.append(
            f'<span title="{nom}" style="display:inline-flex;'
            f'align-items:center;justify-content:center;width:{t}px;'
            f'height:{t}px;border-radius:50%;background:{couleur};'
            f'margin-right:3px;vertical-align:middle;">'
            f'<svg viewBox="0 0 24 24" width="{t}" height="{t}" fill="none" '
            f'stroke="#fff" stroke-width="3" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-label="{nom}" role="img">'
            f'{trace}</svg></span>')
    return "".join(bouts)
