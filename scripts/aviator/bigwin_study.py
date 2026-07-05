"""ÉTUDE DES GROS MULTIPLICATEURS — répond précisément aux questions :
  1. À quelle fréquence tombe un x5 / x10 / x15 / x20 / x50 ? (base rate + attente moyenne)
  2. Les gros se REGROUPENT-ils, ou l'écart entre deux est-il "sans mémoire" (géométrique) ?
  3. Après une SÉCHERESSE de N manches sans gros, un gros devient-il plus probable ? ("dû")
  4. Une SÉQUENCE de multiplicateurs annonce-t-elle un crash précoce (< 1.5x) au coup suivant ?
  5. Les gros clusterisent-ils dans le temps (autocorrélation de l'indicateur "gros") ?

Si tout est plat (P conditionnelle = base rate) et les écarts géométriques => MÉMOIRE NULLE :
impossible de savoir QUAND arrive un gros. C'est le test décisif de la demande utilisateur.
"""
from __future__ import annotations
import sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "aviator.db"


def load(db=DB):
    con = sqlite3.connect(db)
    m = np.array([r[0] for r in con.execute(
        "SELECT multiplier FROM aviator_rounds ORDER BY rowid")], float)
    con.close()
    return m


def gaps_between(indicator):
    """écarts (en nb de manches) entre deux événements True successifs."""
    pos = np.where(indicator)[0]
    return np.diff(pos) if len(pos) > 1 else np.array([])


def run(m):
    n = len(m)
    print("=" * 66)
    print(f"  ÉTUDE DES GROS MULTIPLICATEURS — {n} manches")
    print("=" * 66)
    if n < 60:
        print(f"  ⚠ {n} manches — INDICATIF. Un x10 ~ 1 manche sur 10, un x20 ~ 1 sur 20 :")
        print("    il faut ~2000+ manches pour un verdict solide. Laisse le collecteur tourner.\n")

    # 1. base rates + attente moyenne
    print("  1. FRÉQUENCE & ATTENTE MOYENNE")
    for T in (2, 3, 5, 10, 15, 20, 50):
        p = (m >= T).mean()
        att = (1/p) if p > 0 else float("inf")
        print(f"     ≥ {T:>3}x : {100*p:5.1f}%  ->  en moyenne 1 tous les "
              f"{att:.0f} rounds" if p > 0 else f"     ≥ {T:>3}x : 0 observé")

    # 2. écarts entre gros (>=10x) : géométrique = sans mémoire ?
    print("\n  2. ÉCARTS ENTRE GROS (≥10x) — regroupés ou sans mémoire ?")
    ind = m >= 10
    g = gaps_between(ind)
    if len(g) >= 5:
        p = ind.mean()
        exp_mean, exp_var = 1/p, (1-p)/p**2
        print(f"     {int(ind.sum())} gros | écart réel : moy {g.mean():.1f} (théorie géom {exp_mean:.1f}) "
              f"| var {g.var():.1f} (géom {exp_var:.1f})")
        # ratio var/mean² ~ (1-p) pour géométrique ; index de dispersion
        disp = g.var() / (g.mean()**2) if g.mean() else 0
        print(f"     dispersion {disp:.2f} (géométrique ≈ {1-p:.2f}) -> "
              f"{'CONFORME sans mémoire' if abs(disp-(1-p)) < 0.4 else 'à surveiller (clustering ?)'}")
    else:
        print(f"     seulement {int(ind.sum())} gros ≥10x — pas assez pour l'analyse d'écart.")

    # 3. test du "dû" : P(gros | sécheresse >= d) vs base rate
    print("\n  3. APRÈS UNE SÉCHERESSE, un ≥10x est-il PLUS probable ? (théorie du 'dû')")
    base = (m >= 10).mean()
    since = np.zeros(n, int)
    c = 0
    for i in range(n):
        since[i] = c
        c = 0 if m[i] >= 10 else c + 1
    for d in (3, 5, 10, 15):
        mask = since >= d
        if mask.sum() >= 8:
            pc = (m[mask] >= 10).mean()
            print(f"     sécheresse ≥{d:>2} : P(≥10x) = {100*pc:4.1f}%  (base {100*base:.1f}%) "
                  f"-> {'IDENTIQUE = pas dû' if abs(pc-base) < 0.06 else 'écart (à re-tester)'}")
        else:
            print(f"     sécheresse ≥{d:>2} : trop peu de cas ({int(mask.sum())})")

    # 4. séquence -> crash précoce (< 1.5x) au coup suivant ?
    print("\n  4. UNE SÉQUENCE annonce-t-elle un CRASH PRÉCOCE (<1.5x) au suivant ?")
    base_low = (m < 1.5).mean()
    lo = m < 1.5
    for k in (1, 2, 3):
        # après k crashs bas d'affilée
        cond = np.ones(n - k, bool)
        for j in range(k):
            cond &= lo[j:n-k+j]
        nxt = lo[k:n]
        if cond.sum() >= 8:
            pc = nxt[cond].mean()
            print(f"     après {k} bas (<1.5x) d'affilée : P(bas suivant) = {100*pc:4.1f}% "
                  f"(base {100*base_low:.1f}%) -> {'IDENTIQUE' if abs(pc-base_low) < 0.08 else 'écart'}")
        else:
            print(f"     après {k} bas : trop peu de cas ({int(cond.sum())})")
    # après un GROS, crash bas plus probable ?
    big_prev = (m[:-1] >= 10)
    if big_prev.sum() >= 8:
        pc = (m[1:][big_prev] < 1.5).mean()
        print(f"     juste après un ≥10x : P(bas <1.5x) = {100*pc:4.1f}% (base {100*base_low:.1f}%)")

    # 5. autocorrélation de l'indicateur "gros"
    print("\n  5. CLUSTERING TEMPOREL (autocorr de l'indicateur ≥10x)")
    x = (m >= 10).astype(float); x = x - x.mean()
    acs = []
    for lag in (1, 2, 3, 5):
        if n > lag + 8:
            a, b = x[:-lag], x[lag:]
            r = float((a@b) / (np.sqrt((a@a)*(b@b)) or 1))
            acs.append(r); print(f"     lag {lag} : {r:+.3f}")
    print("\n  VERDICT :")
    flat = base > 0
    print("    Si les fréquences sont stables, les écarts géométriques, et P(gros|sécheresse)=base :")
    print("    -> les gros multiplicateurs sont SANS MÉMOIRE. Impossible de savoir QUAND.")
    print("    La vérité utile : P(≥10x) ≈ {:.0f}% CHAQUE round, indépendamment du passé.".format(100*base))
    print("    « à 13h10 ça crashera à 2.97 » est mathématiquement impossible (provably-fair).")
    print("    Ce qu'on te donne : la PROBA par palier + l'attente moyenne (ci-dessus).")
    print("=" * 66)


if __name__ == "__main__":
    run(load())
