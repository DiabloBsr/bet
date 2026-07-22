"""Lecture des libellés de plages de buts + garde-fou anti-résultat-trop-beau.

Pourquoi ce module existe : un backtest a affiché **+35% de ROI** parce que
`"Le total de buts est supérieur à 4"` était lu comme « exactement 4 ». Un match
à 4 buts était compté gagnant et payé à la cote du « >4 » (~11.0), un événement
bien plus rare. Le libellé est la seule chose qui relie une cote à un résultat :
s'il est mal lu, tout le backtest ment — et il ment dans le sens qui plaît.

Deux protections, à utiliser ensemble :
  parse_goal_range()  -> lecture des libellés, couverte par une table de cas figés
  check_roi()         -> tout ROI positif est SUSPECT tant qu'il n'est pas expliqué

Sur ce site, aucun marché n'est +EV (voir THEORIES_TESTED.md) : les marges
mesurées vont de 5.7% (1X2) à ~20% (score exact). Un backtest qui sort du positif
signale un bug de lecture ou d'appariement, pas une opportunité.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

MAX_GOALS = 12          # plafond dur observé (0 dépassement sur 58 083 matchs : total <= 6)


class SuspectResultError(RuntimeError):
    """Levée quand un backtest produit un ROI positif statistiquement significatif."""


def _norm(s: str) -> str:
    """minuscules sans accents : 'supérieur' et 'superieur' doivent se lire pareil."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def parse_goal_range(label: str, max_goals: int = MAX_GOALS) -> frozenset[int] | None:
    """Ensemble des totaux de buts que ce libellé fait GAGNER, ou None si illisible.

    Renvoyer None est volontaire : mieux vaut écarter une offre que la compter
    à tort. Ne jamais remplacer ce None par un ensemble « par défaut ».

    Deux familles de libellés, à ne pas confondre :
      SEUIL       "> 3.5", "< 3.5", "supérieur à 4", "au moins 3", "3+"
      ÉNUMÉRATION "Le total de buts est de 0, 1 ou 2"
    Un seuil décimal doit être lu comme un nombre, pas comme deux entiers :
    `"> 3.5"` vaut {4,5,...}, surtout pas {3,5}.

    >>> sorted(parse_goal_range("Le total de buts est supérieur à 4"))[:3]
    [5, 6, 7]
    >>> sorted(parse_goal_range("> 3.5"))[:3]
    [4, 5, 6]
    >>> sorted(parse_goal_range("Le total de buts est de 0, 1 ou 2"))
    [0, 1, 2]
    """
    s = _norm(label).replace(",", ".") if re.search(r"\d[.,]\d", _norm(label)) else _norm(label)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None

    ge = ">=" in s or "au moins" in s or "ou plus" in s or bool(re.search(r"\d\s*\+", label))
    le = "<=" in s or "au plus" in s or "maximum" in s
    gt = not ge and (">" in s or "superieur" in s or "plus de" in s)
    lt = not le and ("<" in s or "inferieur" in s or "moins de" in s)

    if ge or gt or le or lt:
        t = max(nums) if (ge or gt) else min(nums)
        entier = float(t).is_integer()
        if ge or gt:
            # v > t  -> t+1 si t entier, sinon ceil(t) ; v >= t -> ceil(t)
            start = int(t) + 1 if (gt and entier) else int(-(-t // 1))
            return frozenset(range(max(start, 0), max_goals + 1))
        # v < t  -> t-1 si t entier, sinon floor(t) ; v <= t -> floor(t)
        end = int(t) - 1 if (lt and entier) else int(t // 1)
        return frozenset(range(0, max(end + 1, 0)))

    if not all(float(n).is_integer() for n in nums):
        return None                    # décimale sans comparateur : illisible, on écarte
    ints = [int(n) for n in nums]
    if "entre" in s and len(ints) >= 2:
        return frozenset(range(min(ints), max(ints) + 1))
    return frozenset(ints)


def check_roi(pnl, context: str = "", raise_on_suspect: bool = True,
              sigmas: float = 3.0) -> str:
    """Verdict sur un backtest. `pnl` = gains par pari (cote-1 si gagné, -1 sinon).

    Un ROI positif à plus de `sigmas` écarts-types n'est pas une découverte : sur
    ce site c'est un bug (mauvaise lecture de libellé, appariement cote/résultat
    décalé, fuite de données). On s'arrête au lieu de rendre un chiffre.
    """
    a = np.asarray(pnl, float)
    n = a.size
    if n == 0:
        return "vide"
    roi, se = a.mean(), a.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    msg = f"ROI {100*roi:+.2f}% +-{100*sigmas*se:.2f} (n={n})"
    if n > 30 and roi - sigmas * se > 0:
        alert = (f"SUSPECT: {context or 'backtest'} -> {msg}. Aucun marche n'est +EV ici "
                 f"(marges 5.7%-20%). Verifier la lecture des libelles (parse_goal_range), "
                 f"l'appariement cote<->resultat, et l'absence de fuite de donnees.")
        if raise_on_suspect:
            raise SuspectResultError(alert)
        return alert
    return msg
