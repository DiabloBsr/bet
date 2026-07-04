# FINDINGS — Bet261 Virtual Football (InstantLeague-8035)

**Clôture de la recherche d'edge sur l'espace historique cotes/résultats — 2026-07-02.**
16 campagnes, ~1 540 cellules testées, discipline constante : split chronologique TRAIN/TEST,
même-signe OOS, test binomial exact, correction BH-FDR, ROI>0 requis sur cotes offertes,
vérification adverse indépendante de tout survivant.

## Verdict global

**Le RNG est calibré au marché à moins de 2 points partout, sans mémoire, et aucune
stratégie systématique historique n'existe.** La cote affichée est la statistique
suffisante ; toute stratégie conditionnelle a une espérance = −marge (5–7 % en 1X2/O-U/BTTS,
10–12 % par partition conjonctive, ~18 % score exact, jusqu'à ~33 % sur les couvertures
chevauchantes Multi-Buts).

**Le book est une GRILLE UNIQUE (preuve arithmétique, campagne 17)** : la somme des 9
HT/FT vaut 1.00000 et toutes les marginales (1X2, Mi-tps 1X2, conjonctifs, totaux/équipe)
s'en déduisent à 0.02–0.19 pp près ; le même outcome 0-0 vendu par 3 instruments porte
une cote strictement identique sur 32 392/32 392 matchs ; zéro arbitrage comptable
(somme 1/cote minimale observée : 1.0981). **Aucune incohérence inter-marchés n'existe** —
le « pricing par modèle séparé », dernière faille théorique, est réfuté.

## Ce qui a été testé (tout NULL sauf mention)

| # | Campagne | Cellules | Résultat |
|---|---|---|---|
| 1 | Signaux RNG globaux (mémoire, lag-1..10, cycles, régimes, drift) | exhaustif | NULL (lag-10 = artefact réfuté) |
| 2 | Audit conditionnel niveau 2 (clean rounds, permutation, MDE) | — | NULL |
| 3 | Équipes : momentum, forme, matchups, biais favori, stratégie ROI | 6 angles | NULL (persistance OOS r=−0.17) |
| 4 | Marchés buts : O/U, BTTS, totals exacts, 9 ligues, vraies cotes | ~900 slices | NULL (ROI = −marge partout) |
| 5 | Arbitrage intra-coupon, cohérence croisée des marchés | 88k events | NULL (0 arb, cohérence 0.025pp) |
| 6 | Fuite API (résultat/seed/provably-fair dans la réponse brute) | — | NULL (rien d'exposé avant clôture) |
| 7 | Position dans le round (hypothèse 3-4-3) | 10 pos | NULL (χ² p=0.32, OOS r=−0.10) |
| 8 | Cote-par-cote 0.05 + chaînes 1/2/3 features (20 lots) | 551 tests | NULL (p-values uniformes) |
| 9 | Trajectoires par équipe : bande×chute, chaînes 2, marges, repos | 221 cellules | NULL |
| 10 | **Fuite-futur** (cotes du match suivant publiées avant le courant, 46 434 cas) | logit+cellules | **NULL — pas de fuite** |
| 11 | Face-à-face : 9 combos V/N/D×V/N/D, chutes croisées, gros cotes | 136 cellules | NULL (1 pseudo-survivant tué en vérif adverse) |
| 12 | Clôture : chaînes buts/BTTS vs cotes offertes, streaks ≥3, quotas de round, répétition de score | 140 cellules | NULL |
| 13 | **TOTALE (17e)** : les 22 marchés × 118 sélections (calibration+ROI) + cohérence inter-marchés (conjonctifs vs grille, 4 prix du 0-0, arbitrage comptable 7 partitions) | 483 cellules | NULL — et preuve de la **grille unique** |

Faux positifs observés au fil des campagnes = exactement le taux attendu par hasard ;
tous tués par la vérification adverse (test miroir, placebo lag-2, bootstrap CI).

## La seule anomalie RNG réelle jamais détectée (inexploitable)

**Répétition de score exact** : une équipe rejoue son score précédent **6.26 %** du temps
vs 5.84 % implicite (p=1.3e-05) ; après double répétition : 7.89 % vs 6.86 % (p=0.017).
Effet réel de +0.4 à +1.1pp — **écrasé par la marge ~18 %** du marché Score exact
(ROI −5 à −14 % dans toutes les cellules). Valeur : contrôle positif prouvant que le
pipeline détecte les vrais effets — ce qui valide les ~1 540 NULL ailleurs. **Ne pas parier.**

Autre effet réel mais pricé : biais buts-par-équipe (Brentford/Brighton over, Everton/
Bournemouth under, t≈8) — intégralement absorbé et même sur-corrigé par le book (gap −2.6pp).

## Plafonds de prédiction (mesurés, OOS)

- Score exact Top-1 ≈ 11.7–12 % · Top-3 ≈ 31.6 % · 1X2 ≈ 55 % · O/U 2.5 ≈ 62 %.
- Aucun des 7 modèles ML (LogReg→XGBoost→stacking) ne bat le log-loss des cotes (0.9601).
- Poids du mélange V2/V5/marché : indifférents (écarts < bruit). Calibration 7×7 :
  +0.3pp Top-1 / −0.4pp Top-3 → double mode (Top-1 calibré, Top-3 brut).

## Doctrine opérationnelle (en vigueur)

1. **Prédire avec V2** (+ trio V2/V5/marché en dashboard) — au plafond, calibré.
2. **Signaler les grosses cotes notables** — informatif, jamais promesse de gain.
3. **Ne jamais miser sur un pattern « vu »** : 16 campagnes prouvent que c'est du biais
   de mémoire sélective (ex. « favori 2.2 en chute gagne 8/10 » → réel : 44 %, ROI −8 %).
4. Suivi forward continu (`trio_tracker.py`) + moniteurs de dérive (`edge_monitor.py`,
   `line_edge_monitor.py`) + refresh hebdo calibration (`refresh_calibration.py`).

## Seules conditions de réouverture de la chasse

1. **Changement de moteur/version** du RNG ou de structure de marge — détecté par les
   moniteurs (résidu global soutenu > 2pp ou ROI OOS > +2 % t>2 en fenêtre récente).
2. **Nouvelle donnée exogène** hors espace historique — ou retour du régime de
   publication anticipée (cf. ci-dessous).

## ⚰️ Post-mortem du mouvement de ligne (2026-07-04)

Le « seul signal positif jamais vu » (ROI +11.5 % à n=182, DOM ; +7.6 % EXT) est
**mort structurellement** : toutes les paires ouverture≠clôture datent du 3-14 juin,
époque où la plateforme publiait les rounds longtemps à l'avance (jusqu'à ~35 h) et
révisait ses cotes. Depuis mi-juin : publication ~17 min avant le coup d'envoi avec
**cotes figées** — 0 changement pré-start sur >100 000 événements en 3 semaines
(1.000 snapshot/event exactement). Le signal ne peut plus ni se confirmer ni
s'exploiter : il n'y a plus de mouvement à suivre. Les moniteurs (line_edge_monitor,
line_paper_trader, 3 signaux DOM/EXT/NUL + CLV) restent armés en SENTINELLES : si la
plateforme repasse en publication anticipée, les paires réapparaîtront d'elles-mêmes.

*Tout re-découpage supplémentaire de la même base ne peut produire que des faux positifs.*
