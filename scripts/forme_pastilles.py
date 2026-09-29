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

FORME_STYLES = {
    "V": ("#21b35a", "✓"),   # victoire : vert, coche
    "D": ("#e8384f", "✕"),   # defaite  : rouge, croix
    "N": ("#c2c8d0", "−"),   # nul      : gris, tiret
}


def pastilles(seq, taille: int = 18) -> str:
    """Une suite « VNDVV » -> pastilles rondes, du plus RECENT au plus ancien.

    Seuls V, N et D sont rendus ; tout autre caractere est ignore. C'est aussi
    ce qui rend l'insertion HTML sure : rien venu de la base ne traverse,
    seulement trois lettres connues.
    """
    bouts = []
    for c in str(seq or ""):
        st = FORME_STYLES.get(c)
        if not st:
            continue
        couleur, glyphe = st
        bouts.append(
            f'<span style="display:inline-flex;align-items:center;'
            f'justify-content:center;width:{int(taille)}px;'
            f'height:{int(taille)}px;border-radius:50%;background:{couleur};'
            f'color:#fff;font-size:{max(int(taille) - 6, 9)}px;'
            f'font-weight:700;line-height:1;margin-right:3px;">{glyphe}</span>')
    return "".join(bouts)
