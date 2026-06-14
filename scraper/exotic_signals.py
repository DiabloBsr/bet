"""EXOTIC SIGNALS — signaux hautes cotes sur marchés dérivés (HT/FT, Mi-temps, Totaux).

Issus de la campagne d'optimisation multi-agents (2026-06-11) :
- 7 mineurs de données × walk-forward 3 fenêtres × vérification adversariale indépendante
- Critique de complétude : seuls les signaux POSITIFS sur les DEUX périodes
  (pré-minage ET post-minage) sont retenus — ROI réaliste = colonne "train".

SIGNAUX RETENUS (ROI train / ROI OOS, cote moyenne) :
  S1  HT/FT 'X/2'  favori home MS_mid     : +32.3% / +36.5%, cote ~14.9  ⭐ le plus stable
  S2  1X2&Total '2 / >3.5'  FS away≥home  : +48.7% / +92.1%, cote ~21.6
  S3  Mi-tps 1X2 '1'  home longshot MS_e  : +22.1% / +60.4%, cote ~6.5, WR ~21%
  S4  Total de buts '1'  home_slight MS_e : +13.2% / +39.8%, cote ~7.7, WR ~16% (volume max)
AJOUTS campagne reverse-engineering moteur (2026-06-12, cf ENGINE_MODEL.md) :
  E1  FTTS '1'  favori home cote ≤1.5     : +6.8% full (p<0.0001, n=1661) / +2.4-4.3% OOS,
      cote ~1.41, WR ~75% — mécanisme PROUVÉ : le moteur booste le favori au 1er but (p=1e-11)
  E2  1X2 '1' favori extrême cote ∈[1.10,1.20] : réel 85.6% vs implicite 81.0%,
      OOS +5.45%, WR ~90% — le marché sous-estime les favoris écrasants
MICRO-STAKES (train à peine positif, cote énorme — mise 0.1-0.2u max) :
  M1  HT/FT '2/1'  favori home MS_mid     : +10.4% / +65.8%, cote ~30
  M2  Total équipe dom '> 3.5'  home_slight cote 5-8 : +12.4% / +77.2%, cote ~7

REJETÉS par le critic (artefacts de fenêtre) : Total ext >3.5 away_slight (-10.8% train),
Total de buts '6' (~0% train), HT/FT '1/X' (loterie 21 wins), PORTFOLIO P1 brut
(contenait des paris mutuellement exclusifs sur 603 matchs).

RÈGLES D'USAGE (recommandations critic) :
- 1 seul pari exotique max par match (jamais '2/1' ET 'X/2' sur le même match)
- Sizing : S1-S4 = 0.3-0.5u ; M1-M2 = 0.1-0.2u (WR 5-19% → séries de 30+ pertes possibles)
- Les signaux segmentés exigent une journée FIABLE (round_info en BDD, pas l'inférence)
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExoticPick:
    signal_id: str       # "S1".."S4", "M1", "M2"
    market: str          # nom du marché extra_markets
    selection: str       # clé de la sélection dans le marché
    cote: Optional[float]  # cote réelle du marché (None si marché absent du payload)
    expected_wr: float   # WR historique train
    roi_train: float     # ROI période pré-minage (l'estimateur réaliste)
    stake: float         # mise recommandée en unités
    reason: str


def _parse_em(extra_markets) -> dict:
    if isinstance(extra_markets, str):
        try:
            extra_markets = json.loads(extra_markets)
        except Exception:
            return {}
    return extra_markets if isinstance(extra_markets, dict) else {}


def _market_odds(em: dict, market: str, selection: str) -> Optional[float]:
    """Cote d'une sélection ; tolère les variations d'espaces dans les clés."""
    m = em.get(market)
    if not isinstance(m, dict):
        return None
    if selection in m:
        v = m[selection]
        return float(v) if isinstance(v, (int, float)) and 1.01 <= float(v) <= 50 else None
    want = selection.replace(" ", "")
    for k, v in m.items():
        if isinstance(k, str) and k.replace(" ", "") == want:
            return float(v) if isinstance(v, (int, float)) and 1.01 <= float(v) <= 50 else None
    return None


def evaluate_exotics(journee: Optional[int], odds_home: float, odds_away: float,
                      extra_markets=None, journee_reliable: bool = True,
                      include_micro: bool = True) -> list[ExoticPick]:
    """Évalue les signaux exotiques pour un match à venir.

    Args:
        journee: numéro de journée (1-38) ; None si inconnue
        journee_reliable: False si la journée vient d'une inférence (les signaux
            segmentés sont alors désactivés, recommandation du critic)
        include_micro: inclure les signaux micro-stakes M1/M2
    """
    picks: list[ExoticPick] = []
    em = _parse_em(extra_markets)
    seg_ok = journee is not None and journee_reliable

    ms_mid = seg_ok and 13 <= journee <= 25
    ms_early = seg_ok and 4 <= journee <= 12
    fs = seg_ok and 34 <= journee <= 38
    home_slight = 1.6 <= odds_home < 2.2 and odds_away >= 2.5

    # S1 — HT/FT 'X/2' : favori home solide MS_mid, nul HT puis le favori PERD.
    # Le marché price ce retournement comme improbable ; il arrive ~9.3%.
    if ms_mid and 1.25 <= odds_home < 1.70:
        cote = _market_odds(em, "HT/FT", "X/2")
        picks.append(ExoticPick(
            "S1", "HT/FT", "X/2", cote, expected_wr=0.093, roi_train=0.32, stake=0.4,
            reason=f"⭐ S1 favori home @{odds_home:.2f} MS_mid → 'X/2' (nul HT puis défaite). "
                   f"ROI +32%/+37% sur 2 périodes, cote ~15",
        ))

    # S2 — 1X2&Total '2 / >3.5' : chaos FS, away non-favori gagne dans un match ouvert.
    if fs and odds_away >= odds_home:
        cote = _market_odds(em, "1X2 & Total", "2 / > 3.5") or _market_odds(em, "1X2 & Total", "2/>3.5")
        picks.append(ExoticPick(
            "S2", "1X2 & Total", "2 / > 3.5", cote, expected_wr=0.094, roi_train=0.49, stake=0.3,
            reason=f"S2 FS + away non-favori @{odds_away:.2f} → '2 ET over 3.5'. "
                   f"ROI +49%/+92%, cote ~22 (chaos fin de saison)",
        ))

    # S3 — Mi-tps 1X2 '1' : home longshot MS_early mène souvent à la pause (~21%)
    # alors que le marché HT le price comme le match entier.
    if ms_early and odds_home >= 3.5:
        cote = _market_odds(em, "Mi-tps 1X2", "1")
        picks.append(ExoticPick(
            "S3", "Mi-tps 1X2", "1", cote, expected_wr=0.21, roi_train=0.22, stake=0.5,
            reason=f"S3 home longshot @{odds_home:.2f} MS_early → mène à la HT 21% du temps. "
                   f"ROI +22%/+60%, cote ~6.5 (meilleure stabilité : WR>20%)",
        ))

    # S4 — Total de buts '1' exact : home_slight MS_early = matchs fermés sous-estimés.
    if ms_early and home_slight:
        cote = _market_odds(em, "Total de buts", "1")
        picks.append(ExoticPick(
            "S4", "Total de buts", "1", cote, expected_wr=0.165, roi_train=0.13, stake=0.4,
            reason=f"S4 home_slight MS_early → exactement 1 but (16.5%). "
                   f"ROI +13%/+40%, cote ~7.7 (plus gros volume de wins)",
        ))

    # E1 — FTTS '1' : le moteur booste le favori home pour le 1er but (LRT p=1e-11)
    # et le marché FTTS ne le price pas (ratio réel/implicite 1.196, p<0.0001, n=1661).
    # NON segmenté → actif même sans journée fiable.
    if odds_home <= 1.50:
        cote = _market_odds(em, "FTTS", "1") or _market_odds(em, "FTTS", "Equipe domicile")
        picks.append(ExoticPick(
            "E1", "FTTS", "1", cote, expected_wr=0.75, roi_train=0.068, stake=0.8,
            reason=f"E1 favori home @{odds_home:.2f} ≤1.5 → marque en 1er ~75% "
                   f"(boost moteur prouvé p=1e-11, marché mal calibré p<0.0001). ROI +6.8%/+2.4-4.3%",
        ))

    # E2 — 1X2 favori extrême [1.10-1.20] : le marché le sous-estime
    # (réel 85.6% vs implicite 81.0% ; backFAV OOS +5.45%, WR 90.5%).
    fav_is_home = odds_home <= odds_away
    fav_cote = odds_home if fav_is_home else odds_away
    if 1.10 <= fav_cote <= 1.20:
        sel = "1" if fav_is_home else "2"
        picks.append(ExoticPick(
            "E2", "1X2", sel, fav_cote, expected_wr=0.87, roi_train=0.054, stake=1.0,
            reason=f"E2 favori extrême @{fav_cote:.2f} ∈[1.10,1.20] → sous-estimé par le marché "
                   f"(réel 85.6% vs implicite 81.0%). OOS +5.45%, WR ~90%",
        ))

    if include_micro:
        # M1 — HT/FT '2/1' : favori home mené à la HT puis renverse. Cote ~30, WR ~4.8%.
        # Exclusif avec S1 sur le même match (jamais les deux : bornes de cote disjointes
        # sauf 1.25-1.70 → S1 prioritaire, M1 seulement si odds_home >= 1.70).
        if ms_mid and 1.70 <= odds_home < 2.0:
            cote = _market_odds(em, "HT/FT", "2/1")
            picks.append(ExoticPick(
                "M1", "HT/FT", "2/1", cote, expected_wr=0.048, roi_train=0.10, stake=0.15,
                reason=f"M1 micro : favori @{odds_home:.2f} MS_mid → comeback '2/1'. "
                       f"ROI +10%/+66%, cote ~30 — MICRO-STAKE",
            ))
        # M2 — Total équipe domicile '> 3.5' : home_slight qui explose (4+ buts perso).
        if home_slight:
            cote = _market_odds(em, "Total equipe domicile", "> 3.5")
            if cote is not None and 5.0 <= cote < 8.0:
                picks.append(ExoticPick(
                    "M2", "Total equipe domicile", "> 3.5", cote,
                    expected_wr=0.19, roi_train=0.12, stake=0.2,
                    reason=f"M2 micro : home_slight → 4+ buts domicile, cote {cote:.2f} ∈ [5;8). "
                           f"ROI +12%/+77% — MICRO-STAKE",
                ))

    return picks
