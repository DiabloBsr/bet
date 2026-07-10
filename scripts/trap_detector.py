"""DÉTECTEUR DE PIÈGE — encode TOUT ce qu'on a prouvé sur Bet261 virtual foot.

Aucun marché n'a d'edge positif (RNG certifié, calibré). Le seul levier est de
PERDRE LE MOINS : viser la marge la plus fine, en pari simple, mises plates.
Ce module note un pari (simple / combiné / panier) et dit à quel point il est
piégeux, pourquoi, et quelle est l'alternative la moins mauvaise.

Constantes = MARGES et ROI OOS **mesurés** sur ~250 000 matchs (split chrono,
IC95). Voir THEORIES_TESTED.md. Module pur (pas de DB) → rapide, déployable.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# marché -> (libellé, marge mesurée, ROI OOS mesuré) ; ROI ≈ -marge (calibré)
MARKETS: dict[str, tuple[str, float, float]] = {
    "1x2":         ("1X2 (fin de match)",       0.057, -0.053),
    "dc":          ("Double Chance",            0.057, -0.053),
    "ou35":        ("Over/Under 3.5 (+/-)",     0.057, -0.056),
    "gng":         ("G/NG (les 2 marquent)",    0.057, -0.053),
    "btts_1h":     ("BTTS 1re période",         0.060, -0.054),
    "multibuts":   ("Multi-Buts (O/U 2.5)",     0.090, -0.092),
    "total_buts":  ("Total de buts (exact)",    0.107, -0.100),
    "mitps_1x2":   ("Mi-tps 1X2",               0.080, -0.070),
    "mitps_dc":    ("Mi-tps DC",                0.080, -0.072),
    "htft":        ("HT/FT (mi-tps/fin)",       0.120, -0.117),
    "score_exact": ("Score exact",              0.240, -0.240),
}

# alternative « canal moins cher » : marché piégeux -> (marché conseillé, raison)
CHEAPER = {
    "multibuts":   ("ou35", "même vue « peu de buts » sur Under 3.5 : marge 5.7% au lieu de 9% (perte /2, prouvé)"),
    "total_buts":  ("ou35", "vise Over/Under 3.5 (marge 5.7%) au lieu du total exact (10.7%)"),
    "score_exact": ("1x2",  "le score exact coûte 24% de marge — passe sur 1X2 / O-U 3.5 (5.7%)"),
    "htft":        ("1x2",  "HT/FT = 12% de marge (le pire du site). Prends le résultat fin de match (5.7%)"),
    "mitps_1x2":   ("1x2",  "le marché mi-temps porte 8% de marge vs 5.7% en fin de match — même pari, moins cher"),
    "mitps_dc":    ("dc",   "la Double Chance fin de match coûte 5.7% vs 8% à la mi-temps"),
}

LONGSHOT_ODDS = 5.0      # au-delà : « grosse cote » = variance forte, aucun value (favori-longshot réfuté)
THIN_MARGIN = 0.060      # <= : le moins mauvais


@dataclass
class Verdict:
    severity: str                    # 🟢 / 🟡 / 🟠 / 🔴
    headline: str
    roi: float                       # ROI attendu (négatif)
    margin: float
    reasons: list[str] = field(default_factory=list)
    better: str | None = None
    expected_loss: float | None = None   # en Ar si mise fournie


def _sev(roi: float) -> str:
    if roi >= -0.06:
        return "🟡"
    if roi >= -0.10:
        return "🟠"
    return "🔴"


def evaluate_single(market: str, odds: float | None = None, stake: float | None = None) -> Verdict:
    """Note un pari SIMPLE. odds sert à détecter la grosse cote (longshot)."""
    if market not in MARKETS:
        return Verdict("⚪", f"Marché inconnu : {market}", 0.0, 0.0,
                       ["Marché non répertorié — impossible d'estimer la marge."])
    label, margin, roi = MARKETS[market]
    reasons = [f"**{label}** : marge mesurée **{margin*100:.1f}%**, ROI attendu **{roi*100:+.1f}%** (calibré, aucun edge)."]
    sev = _sev(roi)
    # modificateur grosse cote
    if odds is not None and odds >= LONGSHOT_ODDS:
        reasons.append(f"🎲 Cote **{odds:g}** = grosse cote : variance très élevée, **aucun value** "
                       "(le biais favori-longshot est réfuté ; l'outsider est même légèrement SURcôté). "
                       "C'est exactement le piège d'un historique à −18.6%.")
        if sev == "🟡":
            sev = "🟠"
    # alternative moins chère
    better = None
    if market in CHEAPER:
        alt_key, why = CHEAPER[market]
        better = f"👉 **{MARKETS[alt_key][0]}** — {why}."
    elif odds is not None and odds >= LONGSHOT_ODDS:
        better = "👉 Un **favori à ~2.0** (même marge 5.7%, variance bien moindre) si tu veux vraiment miser."
    head = {"🟡": "Le moins mauvais (mais tu perds quand même la marge)",
            "🟠": "Cher — marge alourdie",
            "🔴": "PIÈGE — marge lourde"}[sev]
    v = Verdict(sev, head, roi, margin, reasons, better)
    if stake:
        v.expected_loss = -roi * stake
    return v


def evaluate_combo(legs: list[tuple[str, float | None]], stake: float | None = None) -> Verdict:
    """Note un COMBINÉ : les marges se MULTIPLIENT. ROI = ∏(1+roiᵢ)−1."""
    rois, margins, labels = [], [], []
    for m, _o in legs:
        if m in MARKETS:
            labels.append(MARKETS[m][0]); margins.append(MARKETS[m][1]); rois.append(MARKETS[m][2])
    if not rois:
        return Verdict("⚪", "Aucun leg valide", 0.0, 0.0, ["Renseigne des marchés connus."])
    prod = 1.0
    for r in rois:
        prod *= (1 + r)
    roi = prod - 1
    n = len(rois)
    reasons = [
        f"**Combiné {n} legs** : les {n} marges se **multiplient**, elles ne s'additionnent PAS.",
        f"ROI = ∏(1+roiᵢ)−1 = **{roi*100:+.1f}%** (chaque leg ~{sum(rois)/n*100:+.1f}%).",
        f"⚠️ Un combiné {n} legs transforme une perte de ~{-sum(rois)/n*100:.0f}% en **{-roi*100:.0f}%**. "
        "C'est pourquoi le book adore les combinés : sa marge se compose en SA faveur.",
    ]
    v = Verdict("🔴", f"COMBINÉ {n} legs — marge multipliée", roi, 1 - prod, reasons,
                "👉 Casse-le en **paris simples séparés** : l'EV se moyenne au lieu de se multiplier "
                "(bien moins pire, même si toujours négatif).")
    if stake:
        v.expected_loss = -roi * stake
    return v


def evaluate_basket(legs: list[tuple[str, float | None]], stake: float | None = None) -> Verdict:
    """Note un PANIER de simples : l'EV se MOYENNE (≠ combiné). Toujours négatif."""
    rois, has_longshot = [], False
    for m, o in legs:
        if m in MARKETS:
            rois.append(MARKETS[m][2])
        if o is not None and o >= LONGSHOT_ODDS:
            has_longshot = True
    if not rois:
        return Verdict("⚪", "Aucun leg valide", 0.0, 0.0, ["Renseigne des marchés connus."])
    roi = sum(rois) / len(rois)
    reasons = [
        f"**Panier de {len(rois)} simples** : l'EV se **moyenne** (≠ combiné qui multiplie) → "
        f"ROI ≈ **{roi*100:+.1f}%**. C'est mieux qu'un combiné, mais moyenne de négatifs = négatif.",
        "La diversification réduit les **swings**, pas la **marge** : sur la durée tu perds la moyenne.",
    ]
    if has_longshot:
        reasons.append("🎲 Des outsiders à grosse cote **tirent l'EV vers le bas** (−6.4% vs −5.2% pour les "
                       "favoris purs) et ajoutent une variance folle. Le meilleur panier = **favoris purs**.")
    v = Verdict(_sev(roi), f"Panier de {len(rois)} simples", roi, -roi, reasons,
                "👉 Panier de **favoris purs** (~2.0), mises plates : −5.2%, panier gagnant ~47% du temps — "
                "le plus proche du break-even atteignable. Retire les outsiders.")
    if stake:
        v.expected_loss = -roi * stake
    return v


if __name__ == "__main__":
    for m in ("1x2", "htft", "score_exact", "multibuts"):
        v = evaluate_single(m, odds=8.0, stake=10000)
        print(f"{v.severity} {MARKETS[m][0]:<24} ROI {v.roi*100:+.1f}% perte~{v.expected_loss:.0f}Ar/10k")
    print(evaluate_combo([("1x2", 2.0)] * 5).headline)
    print(evaluate_basket([("1x2", 2.0), ("1x2", 2.0), ("gng", 5.0)]).headline)
