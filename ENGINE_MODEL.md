# 🔬 ENGINE MODEL — Le working-flow du moteur Sporty-Tech (ligue 8035)

> Reconstruit le 2026-06-11/12 par campagne de reverse-engineering : 14 facettes,
> ~5 850 matchs, ~17 000 buts, 666 645 sélections de marché évaluées.
> Scripts : `scripts/_wf3_*.py` · Sorties : `exports/wf3_*_final.txt`

---

## 1. LE WORKING-FLOW DU MOTEUR (étape par étape)

### Étape 1 — Calendrier *(confiance : PROBABLE)*
- Round-robin type "cercle de Berger" : 38 journées × 10 matchs, chaque paire 2×/saison.
- La permutation change à chaque saison (PAS de calendrier répété), mais corrélation
  résiduelle faible entre saisons adjacentes (46/508 positions identiques vs 27 attendues, p=0.0004).
- → On ne peut PAS prédire le calendrier futur avec certitude.

### Étape 2 — Probabilités de matchup *(confiance : CERTAIN)*
- **Chaque paire (A,B) a un μ de probabilité quasi-FIXE.**
- Preuve : 99,0 % de la variance des cotes d'ouverture entre occurrences d'une même paire
  = jitter de publication (std 0,036 en logit). Variance vraie résiduelle ≈ 0,000013.
- AUCUN feedback du classement, de la forme ou de la saison sur les probas
  (corr forme→jitter = -0,005, n.s.).

### Étape 3 — Pricing : la grille latente *(confiance : CERTAIN)*
- **Le pricing 1X2 EST une grille Poisson(λh) × Poisson(λa) INDÉPENDANTE pure** :
  écart |p_draw_marché − p_draw_Poisson_fit| = 0,000000 sur 5 859/5 859 matchs.
- λh ∈ [0,42 ; 3,25] (moy 1,635) ; λa ∈ [0,46 ; 2,67] (moy 1,196) ; corr(λh,λa) = −0,67.
- **On peut inverser (λh, λa) EXACTEMENT depuis les cotes 1X2 de n'importe quel match.**
- Tous les marchés dérivent de la même grille latente (identité Score-exact ↔ Total
  vérifiée à 1,00000) + **marge fixe par marché** :

| Marge | Marchés |
|---|---|
| **6,00 %** | 1X2, +/-, G/NG, G/NG extérieur, BTTS 1ère MT |
| 8,00 % | Mi-tps 1X2, Pair/Impair |
| 10,00 % | 1X2 & Total |
| 12,00 % | HT/FT, FTTS, Total de buts, Minute du 1er but, 1X2 & G/NG |
| 15,1-15,3 % | Mi-tps CS, 2ème mi-tps CS |
| 17,8 % | Score exact (mais marge par CELLULE inégale : 2-1/1-2 taxés ~24 %, 2-0/1-0 ~0,5-2,5 %) |
| ~98 % | Multi-Buts (sélections chevauchantes) |

- Cellules cappées à cote 100 : fréquence réelle 0,29 % → **ROI −70 %, interdites.**
- La marge 1X2 est PLATE (6,00 % dans tous les buckets) : pas de favourite-longshot bias structurel.

### Étape 4 — Simulation du résultat *(confiance : CERTAIN sur les déviations)*
Le simulateur NE TIRE PAS depuis la grille de pricing. Déviations systématiques :
1. **Biais offensif** : buts réels 2,953 vs μ pricé 2,831 (**+0,12 but/match**) —
   home 1,700 vs 1,635 ; away 1,254 vs 1,196.
2. **Dépendance négative** type Dixon-Coles (corr résiduelle −0,13, rho fitted ≈ −0,066)
   que le pricing indépendant IGNORE → en réalité :
   - **2-1 : +17 % vs grille (597 obs vs 508, z=+4,13)** · **1-2 : +20 % (475 vs 394, z=+4,24)**
   - 3-3 : −49 % (42 vs 82, z=−4,46) · 4-2, 2-4, 0-2, 2-0, 5-1 : −15 à −30 %
3. **Amplification des écarts** : taux de but du leader boosté (+15,6 % home, +21,1 % away
   quand ils mènent) — les matchs pliés se plient encore plus.
4. **Mode drama** : taux de but 80-90' = 5,66/90min si le match est serré vs 3,26 si plié (**+74 %**).
5. Loi du total ≈ binomiale tronquée N=16 (KL 0,0068), pas Poisson.

### Étape 5 — Timeline des buts *(confiance : CERTAIN)*
- Le moteur tire d'abord le NOMBRE de buts par mi-temps (split ~55,8 % corrélé au 1X2,
  quasi-binomial φ≈0,97), puis place les minutes SÉQUENTIELLEMENT
  (gaps médian 7 min, pas un processus de Poisson temporel).
- Hazard croissant dans chaque mi-temps avec pic à 45'/90'.
- **Boost du favori pour le PREMIER but** au-delà de la composition du score final
  (LRT p=1,2e-11) → le marché FTTS ne le price pas.

### Étape 6 — Publication des cotes *(confiance : CERTAIN)*
- Cotes publiées = μ de la paire + jitter aléatoire (std 0,036 logit).
- 85 % des events multi-snapshots ont des cotes qui bougent ; le close CORRIGE
  partiellement le jitter de l'open (corr jitter↔delta = −0,70).

---

## 2. PLAFOND THÉORIQUE (la réponse au "95 %")

| Cible | Plafond | Pourquoi |
|---|---|---|
| Accuracy 1X2 | **~57 %** (favori) | Le tirage est réellement aléatoire ; les cotes contiennent déjà toute l'info |
| Score exact top1 | ~13-14 % | Idem (grille + déviations DC connues) |
| **Over 0.5** | **~94,6 %** | Seul marché 1X2-adjacent quasi-sûr (cote ~1,03-1,06) |
| DC 1X favori ≤1,20 | ~96 % | cote ~1,06 → ROI +2,3 % OOS |
| **ROI** (la vraie cible) | **+5 à +60 % par edge** | En exploitant les déviations simulateur-vs-pricing ci-dessous |

**Conclusion : viser 95 % d'accuracy 1X2 est physiquement impossible** (le moteur tire
au hasard selon les probas que les cotes donnent). Mais viser un **ROI positif robuste**
en exploitant les ~10 défauts structurels trouvés est réaliste — c'est "penser comme le moteur".

---

## 3. LES EDGES CONFIRMÉS (multi-facettes, classés par solidité)

| # | Edge | Définition | Évidence | ROI |
|---|---|---|---|---|
| 1 | **FTTS favori** | FTTS '1' (home marque en 1er) si cote ≤ 1,5 | margins p<0,0001 (n=1661) + mécanisme timeline p=1e-11 | **+6,8 % full / +2,4-4,3 % OOS** |
| 2 | **Comebacks HT/FT** | X/2 (S1 déjà live), + 1/2 et 2/1 (cotes 13-60) | chains ratios 1,09-1,17 + S1 CONFIRMED campagne 1 | **+9-32 %** |
| 3 | **Total de buts = 1** | parier "exactement 1 but" (cote ~7,9) | crossmarket OOS +3,5 % (n=1758) + S2 campagne 1 +13,4 % | **+3,5-13 %** |
| 4 | **Favori extrême 1X2** | back favori si cote ∈ [1,10-1,20] | draws : réel 85,6 % vs implicite 81,0 %, OOS +5,45 %, WR 90,5 % | **+5,4 %** |
| 5 | **Longshots Mi-tps 1X2** | sélection mi-temps avec p<0,08 (cote ≥ 11) | htft7 P(W≥15)=0,03, +S3 campagne 1 | **+22-60 %** (n petit) |
| 6 | Scores 2-1/1-2 vs grille | sous-cotés +17-20 % par le simulateur | scoregrid z>4,1 FDR | mangé par la marge cellule 24 % → **neutre en direct**, utile pour le predictor V2 |
| 7 | Value-jitter paire | parier quand cote publiée > μ historique paire (EV>0,98) | identity +18,6 % OOS, season, templates (3 facettes convergent, aucune significative seule) | **+5-18 %** → watchlist |
| 8 | Suivre le mouvement open→close | re-scraper avant le round, suivre le drift ≥0,03 logit | templates OOS +11,5 %, season p=0,045 | **+5-11 %** (nécessite scrape tardif) |

## 4. CE QUI RESTE INCONNAISSABLE
- Le tirage RNG lui-même (pas de pattern temporel, pas de régulation intra-round détectée).
- Le calendrier futur exact (permutation par saison).
- La valeur exacte du μ d'une paire neuve (estimable seulement par accumulation).

## 5. RÈGLES D'OR OPÉRATIONNELLES
1. **JAMAIS** parier une cote 100 (cellules cappées, ROI −70 %).
2. **JAMAIS** Multi-Buts (marge ~98 %) ni FTTS longshot (ROI −40 %) ni DC 5-10 (ROI −41 %).
3. Exclure les 473 events de `exports/corrupted_events.json` de toute analyse.
4. La marge minimale est sur : 1X2/G·NG/+- (6 %) — les combos paient 10-12 % de taxe.
5. Tout nouveau signal passe par la watchlist forward avant d'être misé.
