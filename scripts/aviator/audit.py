"""Audit d'ÉQUITÉ Aviator — la distribution réelle des crashs est-elle conforme
à un jeu provably-fair, et quelle est la VRAIE marge maison ?

Modèle théorique (crash fair-scaled) : P(M >= x) = (1 - e) / x  pour x >= 1,
avec e = marge maison. Donc RTP(T) = T * P(M >= T) = (1 - e) pour TOUT seuil T
si le modèle tient (courbe plate = bonne nouvelle). L'écart mesure la marge.

Tests : distribution, marge (RTP), % crash instantané, indépendance des manches
(autocorrélation du log-multiplicateur + test des séquences) car provably-fair => i.i.d.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "aviator.db"


def load(db_path=DB) -> np.ndarray:
    db = sqlite3.connect(db_path)
    m = np.array([r[0] for r in db.execute(
        "SELECT multiplier FROM aviator_rounds ORDER BY rowid")], float)
    db.close()
    return m


def rtp_curve(m, targets=(1.2, 1.5, 2, 3, 5, 10)):
    """RTP(T) = T * P(M>=T). Plat ~ (1-marge) si le modèle fair tient."""
    return {T: float(T * (m >= T).mean()) for T in targets}


def house_edge(m):
    """Marge estimée = 1 - médiane des RTP(T) sur une grille (robuste)."""
    rtps = [T * (m >= T).mean() for T in (1.2, 1.5, 2, 3, 5, 10) if (m >= T).sum() >= 5]
    return 1 - float(np.median(rtps)) if rtps else float("nan")


def independence(m):
    """Provably-fair => manches i.i.d. Autocorrélation log-mult (lag 1..5) + runs test."""
    lm = np.log(np.clip(m, 1.0, None))
    lm = lm - lm.mean()
    ac = {}
    for lag in range(1, 6):
        if len(lm) > lag + 5:
            a, b = lm[:-lag], lm[lag:]
            denom = np.sqrt((a @ a) * (b @ b)) or 1
            ac[lag] = float((a @ b) / denom)
    # runs test sur au-dessus/au-dessous de la médiane
    med = np.median(m)
    s = np.where(m >= med, 1, 0)
    s = s[s != s]  # placeholder
    signs = (m >= med).astype(int)
    runs = 1 + int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    n1, n0 = int(signs.sum()), int((1 - signs).sum())
    if n1 and n0:
        mu = 1 + 2 * n1 * n0 / (n1 + n0)
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        z = (runs - mu) / np.sqrt(var) if var > 0 else float("nan")
    else:
        z = float("nan")
    return ac, float(z)


def report(m):
    n = len(m)
    print("=" * 62)
    print(f"  AUDIT ÉQUITÉ AVIATOR — {n} manches")
    print("=" * 62)
    if n < 20:
        print("  (trop peu de données — laisse le collecteur tourner)")
        return
    print(f"  min {m.min():.2f} | médiane {np.median(m):.2f} | moyenne {m.mean():.2f} "
          f"| max {m.max():.2f}")
    print(f"  crash instantané (=1.00x) : {100*(m<1.005).mean():.1f}%")
    print("\n  Survie P(M>=x) — réel vs fair(3%) :")
    for x in (1.5, 2, 3, 5, 10, 20):
        fair = 0.97 / x
        print(f"    >= {x:>4}x : réel {100*(m>=x).mean():5.1f}%   fair {100*fair:5.1f}%")
    print("\n  RTP(T) = T·P(M>=T)  [plat ~ (1-marge) si équitable] :")
    for T, v in rtp_curve(m).items():
        print(f"    T={T:>4} : RTP {100*v:5.1f}%")
    e = house_edge(m)
    print(f"\n  >>> MARGE MAISON estimée : {100*e:.1f}%  (Spribe annonce ~3%)")
    ac, z = independence(m)
    print(f"\n  Indépendance (provably-fair => i.i.d.) :")
    print(f"    autocorr log-mult : " + " ".join(f"lag{k}={v:+.3f}" for k, v in ac.items()))
    print(f"    runs test z = {z:+.2f}  ({'OK i.i.d.' if abs(z) < 2 else 'anomalie ?'})")
    verdict = "CONFORME (équitable, marge ~loi)" if abs(e - 0.03) < 0.05 and abs(z) < 2.5 \
        else "à surveiller (échantillon/écart)"
    print(f"\n  VERDICT : {verdict}")
    print("  ⚠️ Le crash est provably-fair : IMPRÉVISIBLE. Cet audit mesure l'équité,")
    print("     il ne prédit RIEN. Aucune stratégie ne bat la marge (cf. simulateur).")
    print("=" * 62)


if __name__ == "__main__":
    report(load())
