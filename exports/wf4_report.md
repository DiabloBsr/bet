# RAPPORT DE SYNTHÈSE WF4 — Campagne multi-ligues (9 ligues, vérification adversariale complète)

> Date : 2026-06-12 · Périmètre : 8035 (En, ~2 semaines) + 8 nouvelles ligues (8036 It, 8037 Es,
> 8042 Fr, 8043 De, 8044 Pt, 8056 CL, 8060 CAN, 8065 CdM, <48 h de données).
> 14 domaines analysés, 17 findings confirmés après attaque adversariale, ~733 tests comptés
> dans le domaine le plus scanné. Volumes mesurés sur le premier jour complet 9-ligues
> (2026-06-12) : `exports/wf4_synthese_volumes.json` (script `scripts/_wf4_synthese_1.py`).

---

## 0. LE MESSAGE CENTRAL (à lire avant tout)

**Il ne reste qu'UN SEUL edge jouable confirmé** : FTTS « 1 » sur favori domicile ≤ 1,50,
championnats uniquement, cote moyenne **1,37**, ROI corrigé **+4,9 %**.

**Dans la zone cible de l'utilisateur (cote ≥ 1,6), la campagne n'a RIEN trouvé — et c'est
un résultat fort, pas un échec de puissance** : calibration 1X2 plate 1,8-5,0 (54 buckets,
zéro cellule p ≤ 0,01), totals morts, BTTS efficient, HT/FT imbattable à l'ouverture,
combos toxiques, scores exacts tous négatifs (213 cellules scannées, 0 positive).
La valeur de la campagne est donc : (a) un edge volume-énorme à 1,37, (b) une liste
d'interdictions qui économisent 5-18 % de marge sur des milliers d'occasions/jour,
(c) des règles structurelles pour le predictor V2, (d) une watchlist v3 resserrée dont
3 candidats à cote ≥ 1,6 hérités de la campagne précédente restent à statuer.

---

## 1. VOLUMES MESURÉS (jour 2026-06-12, 9 ligues actives)

| Mesure | Valeur/jour |
|---|---|
| Events listés avec cote d'ouverture (9 ligues) | ~24 350 |
| — dont championnats (8035 + 5 nouveaux) | ~15 000 |
| — dont coupes (8056/8060/8065) | ~9 300 |
| Déclencheurs FTTS (home ≤ 1,50, championnats) | **~3 366** (23,2 % des events championnat) |
| Zone E2 à ÉVITER (favori [1,10-1,20)) | ~1 081 |

Rythme : un round de 9-12 matchs toutes les ~2 min par ligue, 24 h/24. Le volume n'est
jamais la contrainte ; la contrainte est la fenêtre de capture pré-kickoff (cf. §2.1).

---

## 2. CLASSEMENT DES FINDINGS PAR ROI ATTENDU × VOLUME (top 8)

Critère : impact économique = ROI corrigé (verdict adversarial) × occasions/jour.
Un seul finding a un ROI positif ; les suivants sont classés par **perte évitée** ou
par impact structurel sur le predictor V2.

### #1 — FTTS « 1 » favori domicile, championnats uniquement — LE seul edge jouable
- **Règle d'exécution exacte (GELÉE)** : si la cote 1X2 domicile à l'OUVERTURE (snapshot
  MIN(id)) est ≤ 1,50, parier « FTTS = 1 » (domicile marque en premier) à la cote
  d'ouverture du MÊME snapshot. **Ligues autorisées : 8035, 8036, 8037, 8042, 8043, 8044.
  INTERDIT en coupes (8056, 8060, 8065)** : à cotes appariées, champs +6,8 % (p=0,005)
  vs coupes −0,2 % ; coupes pooled n=1372 → −0,4 %. Settlement : premier but via
  goals_json (équipe du but de minute min) ; « Pas de but » (0-0) = perdu.
- **Chiffres** : cote moy **1,369** · WR **77,4 %** · ROI corrigé **+4,9 %**
  (recalcul adversarial n=1959, p=1,8e-4, IC95 bootstrap [+2,3 ; +7,4]) · marge marché 12 %.
- **Volume** : ~3 366 déclencheurs/jour → EV théorique max ≈ **+165 u/jour** à 1 u flat.
  Réalisme : la plupart des snapshots d'ouverture sont capturés 0/+1 min autour du kickoff ;
  le sous-ensemble strictement pré-kickoff fait **+6,39 %** (n=550, p=0,008). En ne misant
  que ce qui est capturable pré-kickoff (~1/3 du flux actuel), EV ≈ **+50-70 u/jour**.
- **Réserves (dimensionner petit)** : holdout forward pur +2,92 % n.s. (462 paris) ;
  dernier tercile +2,8 % n.s. ; 8036/8042 retombés ~0 % avec plus de data. Confiance 0,80.
  → Mise flat 1 u, kill-switch watchlist (cf. proposition v3).
- **Combinaison sans double-compte** : ce finding EST E1 (même règle, périmètre restreint
  aux championnats) → **remplacer E1 par cette version, ne pas cumuler**. Un match qui
  déclenche à la fois un pick TIER1 1X2 home (cote ≤ 1,5) et le FTTS est le MÊME risque
  corrélé (le 1er buteur prédit le vainqueur) : **1 seul pari par match**, priorité FTTS
  (12 % de marge mais +4,9 % net vs 1X2 ≈ −marge).

### #2 — Cap dur du simulateur : total ≤ 6 buts (structurel, confiance 0,98)
- **Règle** : predictor V2 → tronquer toute grille Poisson à total ≤ 6 et score équipe ≤ 6,
  puis renormaliser (2,67 % de masse brute sur des scores impossibles). Ne JAMAIS acheter
  une sélection dont le payoff dépend de 7+ buts (ROI −100 % déterministe).
- 57 383 résultats, zéro 7+ ; le book price déjà la troncature (sélection « 6 » à 0,94×
  grille tronquée). Volume : toutes les grilles calculées, tous les jours.

### #3 — ANTI-EDGE : favori extrême [1,10-1,20) MORT (E2 à retirer, confiance 0,93)
- **Règle** : ne plus JAMAIS backer le favori 1X2 coté [1,10-1,20) à l'ouverture.
  Nouvelle ère : parfaitement calibré (wr 0,802 vs devig 0,819, p=0,20), ROI **−7,57 %**
  (n=847, p=1,7e-6, IC95 [−10,7 ; −4,5], survit Bonferroni ×733). Identique en clôture.
- **Volume évité : ~1 081 occasions/jour** → ≈ **80 u/jour de perte évitée** si on suivait
  encore E2. Impact systèmes : retirer E2 de la watchlist ET recalibrer `tier1_picker.py`
  (TIER_1_ULTRA = cote ≤ 1,30 vit exactement dans cette zone : son WR « attendu 82 % »
  n'a plus de coussin, l'espérance y est −marge).

### #4 — RETRAIT des 5 edges de segment du STRATEGY_REPORT (confiance 0,93)
- **Règle** : retirer de `strategy_engine.py` (tables `COTE_EDGES`/segments DS/MS_*) les
  règles P1-P5 (claims +19,6 à +25,9 %). En walk-forward + réplication : P1 −14,3 %,
  P2 +0,56 % (p=0,94), P3 +2,6 %, P4 −3,9 %, P5 −11,5 %. Artefacts de full-sample
  (grilles segments × buckets sans correction, n=14-100).
- Volume : chaque pick que le strategy engine aurait émis = espérance −4 à −14 %.

### #5 — Blacklist TOTALS : Over 3.5, Under 3.5, « Total de buts = 1 » (confiances 0,95)
- **Over 3,5** : −7,0 % corrigé (n=22 107), 90+ segments scannés, zéro positif. Le biais
  offensif +0,12 est RÉEL mais déjà intégralement pricé (le book price l'O/U depuis son
  simulateur, pas depuis la grille 1X2, et met quasi toute la marge 6 % sur l'Over).
- **Under 3,5** : −5,8 % blanket ; gros favoris (« drama-immunisés ») −6,8 % — faux refuge.
- **« Total = 1 » (edge historique #3, +3,5-13 % revendiqué) : MORT** : −8,2 % corrigé
  (8035 full p=0,004, pooled-9 −9,6 %, 9 ligues sur 9 négatives). L'ancien +3,5 % OOS
  (n=1758) = bruit. Toutes les 7 sélections exactes négatives (marge 12 %).
- Volume concerné : la totalité du slate (~24 000 events/jour) → c'est la plus grosse
  économie de la campagne en valeur absolue si un système misait des totals.

### #6 — Blacklist BTTS et combos (confiances 0,80-0,95)
- **G/NG = marché efficient** : pricé depuis le simulateur (biais +0,12 et Dixon-Coles
  inclus), marge 6 % plate. Oui −5,2 %, Non −6,0 % corrigés. Zéro poche OOS.
- **Pire poche** : BTTS Non quand |λh−λa| ∈ [0,6 ; 1,3) → **−8,3 %** corrigé
  (~35 % des events ≈ 8 400 occasions/jour à éviter).
- **Combos 1X2&G/NG (marge 12 %)** : les 6 sélections entre −8,9 et −19,3 % ;
  « X & aucun but » = **−17,9 %** corrigé, pire pari mesuré du marché. Le boost
  simulateur 2-1/1-2 ne survit à aucune marge de combo.

### #7 — Ratios de grille NEW-ERA pour le predictor V2 (confiance 0,78)
- **Règle** : remplacer les déviations ENGINE_MODEL calibrées sur 8035-old (0-0 −36 %,
  draws −7 %, 3-3 −49 % : contaminées par l'audit de corruption, ère ancienne uniquement)
  par les ratios new-era poolés (n=13 289) :
  `1-0 : 0,721 · 0-1 : 0,722 · 0-0 : 0,866 · 2-0 : 0,868 · 0-2 : 0,846 · 1-1 : 1,079 (n.s. après Bonferroni) · 2-1 : 1,188 · 1-2 : 1,224 · 2-2 : 1,311 · 3-3 : ≈1 (n.s.) · DRAWS : ≈1 (n.s.)`
- Aucun pari direct : les 213 cellules Score exact/Total au prix offert sont TOUTES
  négatives (le boost 2-1/1-2 est mangé par la marge cellule ~24 %). Usage : predictor V2,
  settlement, détection de value future.

### #8 — Lois d'indépendance + hygiène data (confiances 0,80-0,93)
- **Intra-round** : matchs d'un même round statistiquement INDÉPENDANTS (CI95
  équicorrélation [−0,0055 ; +0,0061], power check validé). Aucun bet conditionnel
  intra-round ; les combos multi-matchs d'un round pricent comme des produits — pas de
  piège, pas d'edge. **Round N → N+1** : stratégies conditionnelles mortes ET
  structurellement non bettables (résultats du round N publiés ~2 min APRÈS le kickoff
  de N+1).
- **Séquences/forme** : le sur/sous-régime récent vs cotes ne prédit RIEN (LRT poolé-9
  p=0,35 ; coefs de signe opposé entre ligues). Domaine clos — confirme « les cotes
  contiennent toute l'info ».
- **DEDUP OBLIGATOIRE** (correction adversariale) : ~492 events clones, **100 % en 8035**
  (zéro dans les nouvelles ligues), pas couverts par corrupted_events.json. Règle correcte :
  même (competition, team_a, team_b) + MÊMES cotes d'ouverture + MÊME goals_json non-nul
  → garder MIN(id), quel que soit l'écart temporel (le seuil < 30 min du finding initial
  supprime 60 % de faux positifs et rate 6 paires).

---

## 3. COMBINAISON AVEC LES ACQUIS EXISTANTS (E1, E2, S1-S4, TIER1) — sans double-compte

| Acquis | Statut après campagne | Action |
|---|---|---|
| **E1** (FTTS favori ≤ 1,5) | **REMPLACÉ** par le finding #1 (périmètre championnats, jamais coupes) | Mettre à jour la définition ; un seul pari par match déclencheur |
| **E2** (favori [1,10-1,20]) | **MORT** (−7,57 %, calibré parfait) | Retirer de la watchlist et de tout système ; ne pas « compenser » via TIER_1_ULTRA qui mise la même zone |
| **S1** (X/2 favori home, HT/FT) | Réfuté par 2 findings du domaine htft **mais sans verdict adversarial** | SUSPENDRE les mises, statuer en watchlist v3 (slot gelé, 0 mise) |
| **S2** (Total = 1) | **MORT** (= edge #3, −8,2 % corrigé) | Retirer définitivement |
| **S3** (longshots Mi-tps) | Non testé cette campagne (généralisé en `mitps_longshot_global`) | Conserver en watchlist v3 — c'est LE candidat cote ≥ 1,6 (cote ≥ 11) |
| **S4 / edges 7-8** (value-jitter paire, follow drift) | Non statués cette campagne | Conserver en watchlist v3 ; testables désormais sur 9 ligues |
| **TIER1 picker** | WR « attendus » calibrés sur l'ère old + zone E2 | Recalibrer : espérance 1X2 à cote ≤ 1,7 = −marge partout (finding #5 mid-odds : H −4,3 / D −7,4 / A −5,7 %) ; TIER1 reste un outil d'accuracy, PAS de ROI |
| **strategy_engine COTE_EDGES / segments** | **ARTEFACTS** (finding #4) | Retirer P1-P5 et toute règle segment × bucket dérivée du full-sample |
| **watchlist_registry.json** (fade_serie, sous_regime, standings_*) | Le null séquentiel + « fondamentales = zéro info » les condamnent | Purger dans la v3 (cf. proposition) |

Règle anti-double-compte générale : **un event = un seul ticket**. Tous nos signaux
positifs vivent sur le même déclencheur (favori domicile fort) ; empiler FTTS + 1X2 +
combos sur le même match multiplie la variance sans ajouter d'espérance (les combos
re-paient la marge 10-12 %).

---

## 4. CONTRADICTIONS ENTRE DOMAINES (et arbitrages rendus)

1. **ENGINE_MODEL §3 edge #3 (« Total = 1 », +3,5-13 %) vs domaine totals (−8,2 %)** :
   contradiction frontale, tranchée par n=7 691 + pooled-9 et mécanisme (le book price
   les totaux exacts depuis le simulateur ; on paie la marge 12 %). ENGINE_MODEL à amender.
2. **ENGINE_MODEL edge #4 / E2 (+5,45 % OOS) vs anti-finding E2 (−7,57 %)** : tranchée —
   l'edge n'existait que sur 8035-old et n'était déjà que +0,96 % à l'ouverture. Mort.
3. **Titre du verdict poolabilité (« tout edge structurel se transpose ») vs findings FTTS
   (champs only) et E2 (mort)** : le vérificateur a corrigé — seules les propriétés de
   CALIBRATION se transposent (marge 6 %, grille, cap 6, ratios DC), **pas les edges**.
   Chaque edge doit être re-prouvé par famille de ligues (championnat vs coupe).
4. **ENGINE_MODEL étape 4 (déviations 0-0 −36 %, 3-3 −49 %, draws −7 %) vs ratios new-era
   (0-0 0,866, 3-3 n.s., draws n.s.)** : l'ancienne calibration est contaminée par
   l'exclusion d'audit (125/473 corrompus affichés 0-0, ère old uniquement). Arbitrage :
   ratios new-era pour tout usage futur ; les edges « fade 0-0 / déficit de nuls » de
   l'ère old sont surestimés.
5. **« Comebacks HT/FT sous-pricés » (acquis) vs domaine htft (« comebacks SUR-pricés aux
   cotes offertes »)** : non statué adversarialement (finding rejeté « no verdict ») mais
   converge avec la réfutation S1. La déviation simulateur est réelle, la cote offerte la
   sur-paie. À trancher en watchlist (0 mise en attendant).
6. **BTTS « Oui sous-pricé gap [0,6-1,3) » (rejeté, non jouable) vs « Non = pire pari même
   zone » (confirmé)** : pas une contradiction — même phénomène vu des deux côtés ; le
   vérificateur a dégradé l'excès au-delà de la marge (p=0,076) : zone réelle mais
   inexploitable côté Oui, à blacklister côté Non.
7. **Amplification des écarts / drama mode (ENGINE_MODEL) vs « 1/>3,5 » et Under favoris
   tous négatifs** : les déviations du simulateur sont réelles MAIS toutes pricées dans
   les marchés dérivés. La seule déviation NON pricée qui survit = le boost du favori pour
   le 1er but (FTTS), et seulement en championnat.
8. **Dedup : « nuit multi-ligues, toutes ligues » (claim) vs « 100 % 8035, signature
   cotes+goals_json » (vérif)** : arbitrage = règle signature (cf. #8), le seuil 30 min
   est faux.
9. **Intra-round NULL vs anomalie lag-1 8036 (r=+0,28, p=0,0016)** : hors périmètre du
   null (cross-round, une ligue, multiplicité) ; piste de surveillance non-bettable, ne
   réfute pas l'indépendance intra-round.

---

## 5. CE QUE LA CAMPAGNE A DÉFINITIVEMENT FERMÉ

- 1X2 mid-odds 1,8-5,0 : plat, = −marge (zéro cellule sur 54 ; les « presque positives »
  flippent en split temporel).
- O/U 3,5 (les deux côtés), team totals, 1X2&Total, totaux exacts : tous négatifs.
- G/NG toutes zones + combos 1X2&G/NG + BTTS 1ère MT : négatifs.
- HT/FT à l'ouverture (9 issues), X/2 inclus (sous réserve verdict, cf. §4.5).
- Edges segment DS/MS (strategy_engine), cycle saisonnier favori (artefact de données
  cassées en fin de saison), conditionnels intra-round et inter-rounds, sur/sous-régime,
  features fondamentales (déjà 0/22 en campagne précédente).
- E2 favoris extrêmes ; « Total = 1 » ; déficit de nuls / 0-0 de l'ère old.

## 6. PROCHAINES ÉTAPES RECOMMANDÉES (par priorité)

1. **Mettre en production surveillée le FTTS championnats** (mise flat 1 u, plafond
   d'exposition, kill-switch watchlist) en chassant la fenêtre pré-kickoff : améliorer le
   scraper pour capturer les cotes AVANT le coup d'envoi (l'edge y est +6,4 %).
2. **Purger les systèmes** : E2, S2, COTE_EDGES/P1-P5, anciens ratios de grille ;
   recalibrer TIER1 (accuracy ≠ ROI) ; implémenter le dedup signature + cap ≤ 6.
3. **Statuer les 3 candidats cote ≥ 1,6 hérités** (mitps_longshot ~11+, value_jitter,
   follow_drift) sur les 9 ligues — c'est la seule voie crédible vers l'objectif
   utilisateur « ROI à cote élevée » ; le drift nécessite un re-scrape tardif (edge #8).
4. **Relancer l'audit de corruption sur les 9 ligues** (le mécanisme a disparu du pipeline
   le 06-12, mais l'audit n'a couvert que 8035) et re-estimer les ratios new-era quand les
   nouvelles ligues auront ≥ 1 semaine de données (le 1-1/2-1 bouge encore, z~3).
5. **Watchlist v3** : adopter la proposition `exports/wf4_watchlist_v3_proposal.md`
   (définitions gelées, promotion z ≥ 2 / n ≥ 80 / ROI > 0, kill-switch symétrique).

---
*Sources : findings vérifiés adversarialement (scripts `scripts/_wf4_*.py`, exports
`exports/wf4_*.json`), ENGINE_MODEL.md, STRATEGY_REPORT.md, data/watchlist_registry.json,
volumes `exports/wf4_synthese_volumes.json`. Aucun fichier existant modifié.*
