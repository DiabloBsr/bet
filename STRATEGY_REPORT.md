# 📊 RAPPORT STRATÉGIQUE — Analyse profonde 4 543 matchs (J1-J37)

## 🎯 Segmentation de saison

| Segment | Journées | n matchs |
|---|---|---|
| **DS** (Début) | J1-J3 | 446 |
| **MS_early** | J4-J12 | 1 097 |
| **MS_mid** | J13-J25 | 1 617 |
| **MS_late** | J26-J33 | 988 |
| **FS** (Fin) | J34-J38 | 395 |

---

## 📊 Phase 1 : Base rates par segment

| Métrique | DS | MS_early | MS_mid | MS_late | **FS** |
|---|---|---|---|---|---|
| **1 (home win)** | 48.0% | 48.1% | 47.7% | 48.5% | **49.1%** |
| **X (draw)** | **19.7%** | 22.7% | 21.8% | 22.7% | **25.1%** |
| **2 (away win)** | 32.3% | 29.2% | 30.5% | 28.8% | **25.8%** |
| Buts moyens | 3.00 | 2.99 | 2.94 | 2.94 | 3.01 |
| **HT X (nul mi-tps)** | 41.3% | 42.3% | 42.3% | 40.9% | **45.8%** |
| BTTS OUI | 55.8% | 59.2% | 58.1% | 58.5% | 56.5% |

**Insights :**
- ⭐ **FS** = +25% de nuls (vs DS 19.7%) — finale prudente
- ⭐ **DS** = moins de nuls (plus de "matches décidés")
- 🔥 **FS HT_X = 45.8%** : presque la moitié des 1ères mi-temps en FS sont nulles
- Buts moyens stables (~3.0) tous segments → modèle Poisson stable

---

## 🔄 Phase 2 : Transitions HT → FT

| Transition | DS | MS_mid | **FS** |
|---|---|---|---|
| **HT=1 → FT=1** | **86%** | 81% | **77%** |
| HT=X → FT=1 | 38% | 37% | **44%** ⭐ |
| HT=X → FT=X | 30% | 33% | 34% |
| HT=X → FT=2 | 32% | 30% | 22% |
| **HT=2 → FT=2** | **78%** | 67% | **64%** |

**Insights stratégiques majeurs :**
- 🥇 **DS** = matches décisives : si HT=1 → 86% chance FT=1
- 🥇 **FS** = **HT_X → FT_1 à 44%** ! Si match nul à la pause en FS, parier home gagnant value énorme
- 🥈 **2nd mi-temps plus offensive** : 1.62-1.72 buts vs 1.29-1.36 en 1ère
- ⚠️ FS = remontées : HT=1 → FT=1 baisse à 77% (vs 86% en DS)

---

## 🌟 Phase 3 : Forces des équipes par segment

### 🔥 Équipes qui PEAK par segment (Δ ≥ +10pp WR home)

| Segment | Équipes peakers (Δ WR home) |
|---|---|
| **DS** | **West Ham +20pp** (62% WR), London Blues +16pp (68%) |
| MS_early | Bournemouth +10pp (51%) |
| MS_mid | *aucune* |
| MS_late | *aucune* |
| **FS** | **Spurs +23pp** (76% WR!), **C. Palace +21pp** (73%), London Blues +13pp (65%), Sunderland +11pp (26%) |

### ❄️ Équipes qui CHUTENT par segment (Δ ≤ -10pp)

| Segment | Équipes droppers |
|---|---|
| **DS** | **Leeds 5% WR home (-15pp!)**, Fulham -14pp, Brentford -13pp |
| MS_early | West Ham -14pp |
| MS_mid | *aucune* |
| MS_late | *aucune* |
| **FS** | **Everton 18% WR home (-17pp!)**, A. Villa -11pp |

---

## 🎰 Phase 4 : Paires GOLD par segment

| Segment | PAIRES OR HOME | SCORE COMBO GOLD | PAIRES TRAP |
|---|---|---|---|
| DS | 0 (échantillon trop court) | 0 | 0 |
| **MS_early** | 11 | 22 | 10 |
| **MS_mid** | 15 | **67** | 10 |
| MS_late | 7 | 11 | 3 |
| FS | (trop court) | (trop court) | (trop court) |

→ Le segment **MS_mid** est où le marché est le plus mispriced (67 paires SCORE COMBO GOLD!).

---

## 💰 Phase 5+6 : Edges par cote × segment (LES GROSSES TROUVAILLES)

### 🔥 EDGES POSITIFS (Buckets profitables)

| Segment | Bucket | Side | n | WR réel | Implicite | EDGE | **ROI** |
|---|---|---|---|---|---|---|---|
| **DS** | Cote [2.20-2.70) "équilibré" | home | 67 | 49.3% | 41.1% | **+8.2pp** | **+19.8%** |
| **DS** | Cote [1.50-1.80) "favori modéré" | **away** | 33 | 75.8% | 61.3% | **+14.4pp** | **+23.5%** |
| **DS** | Cote [3.50-5.00) "underdog" | **away** | 100 | 30.0% | 24.1% | **+5.9pp** | **+24.5%** |
| **DS** | Cote [1.30-1.50) "favori solide" | away | 14 | 78.6% | 72.1% | +6.5pp | +9.0% |
| **MS_early** | Cote [5+) "long shot" | **home** | 115 | 18.3% | 12.7% | **+5.6pp** | **+43.9%** |
| **MS_early** | Cote [2.70-3.50) "non-favori léger" | home | 54 | 38.9% | 32.9% | +6.0pp | +18.1% |
| **MS_mid** | Cote [5+) "long shot" | home | 152 | 15.8% | 12.5% | +3.2pp | +25.9% |
| **MS_mid** | Cote [5+) "long shot" | away | 488 | 14.5% | 12.2% | +2.4pp | +19.6% |
| **MS_late** | Cote [5+) "long shot" | home | 99 | 16.2% | 12.6% | +3.6pp | +28.4% |
| **MS_late** | Cote [1.30-1.50) "favori solide" | **away** | 27 | 77.8% | 71.5% | +6.2pp | +8.7% |
| **FS** | Cote [5+) "long shot" | **home** | 42 | 19.0% | 12.3% | **+6.8pp** | **+55.4%** |
| **FS** | Cote [3.50-5.00) "underdog" | home | 30 | 30.0% | 24.5% | +5.5pp | +22.6% |
| **FS** | Cote [1.30-1.50) "favori solide" | away | 14 | 78.6% | 72.0% | +6.6pp | +9.2% |

### ❄️ TRAPS À ÉVITER ABSOLUMENT

| Segment | Bucket | Side | n | WR réel | EDGE | **ROI** |
|---|---|---|---|---|---|---|
| **DS** | Cote [1.80-2.20) "léger favori" | home | 84 | 38.1% | -12.7pp | **-25.0%** |
| **DS** | Cote [3.50-5.00) "underdog" | home | 34 | 8.8% | -15.5pp | **-63.7%** |
| MS_early | Cote [1.80-2.20) "léger favori" | away | 85 | 34.1% | -16.9pp | -33.1% |
| MS_mid | Cote [1.00-1.30) "favori extrême" | home | 143 | 72.0% | -10.1pp | -12.3% |
| MS_late | Cote [1.30-1.50) "favori solide" | home | 146 | 63.7% | -8.2pp | -11.4% |
| **FS** | Cote [1.50-1.80) "favori modéré" | **away** | 35 | 34.3% | **-26.5pp** | **-43.6%** |
| **FS** | Cote [1.80-2.20) "léger favori" | away | 25 | 32.0% | -18.5pp | -36.7% |
| **FS** | Cote [2.20-2.70) "équilibré" | home | 56 | 28.6% | -12.7pp | -30.8% |

---

## ⏰ Phase 7 : Minute du premier but

| Métrique | DS | MS_early | MS_mid | MS_late | FS |
|---|---|---|---|---|---|
| % sans but | 3.0% | 2.6% | 3.0% | 2.1% | 2.1% |
| Médiane (min) | 27 | 27 | 27 | 27 | 28 |
| % 1er but ≤ 15' | 23.6% | 23.6% | 23.6% | 22.7% | 23.4% |
| % 1er but ≤ 30' | 59.8% | 59.8% | 57.3% | 58.0% | 55.0% |
| % 1er but ≤ 45' (HT) | 78.1% | 78.1% | 76.8% | 76.2% | 75.0% |

**Insights :**
- Constant tous segments (~23% goal ≤ 15', ~78% goal en 1ère mi-temps)
- **Over 0.5 1ère mi-temps : ~77% toutes saisons** → pari très sûr
- Pic d'activité : **15-30'** (33-36% des 1ers buts dedans)

---

## 🎯 STRATÉGIES OPÉRATIONNELLES par segment

### 🥇 DS (J1-J3) — "Saison qui démarre"
**Picks GOLD :**
1. 🔥 **Away favori modéré [1.50-1.80)** : ROI +23.5%
2. 🔥 **Away underdog [3.50-5.00)** : ROI +24.5% — upsets fréquents
3. 🔥 **West Ham home** : +20pp delta — value bet récurrent
4. 🔥 **London Blues home** : +16pp delta
5. 🔥 HT=1 → FT=1 confiance 86%

**TRAPS DS :**
- ❌ Home léger favori [1.80-2.20) : ROI -25%
- ❌ Home underdog [3.50-5.00) : ROI -63.7%! (jamais parier home @3.50-5)
- ❌ Leeds home (5% WR! -15pp)
- ❌ Fulham home (-14pp)
- ❌ Brentford home (-13pp)

### 🥈 MS_early (J4-J12) — "Mise en place"
**Picks GOLD :**
1. 🔥 **HOME long shot [5+]** : ROI +43.9% ! (signal le plus rentable)
2. 🔥 Home non-favori [2.70-3.50) : ROI +18.1%
3. Bournemouth home : +10pp

**TRAPS :**
- ❌ Away léger favori [1.80-2.20) : ROI -33%
- ❌ Home équilibré [2.20-2.70) : ROI -19%

### MS_mid (J13-J25) — "Cœur de saison"
- 🟡 Marché plus efficient
- 🔥 Long shots (home + away) toujours value : +20-26% ROI
- ❌ Favoris extrêmes home [1.00-1.30) : -12% ROI (cote trop basse)

### MS_late (J26-J33) — "Sprint final"
- 🔥 Home long shot : ROI +28%
- 🔥 Away favori solide [1.30-1.50) : ROI +9%
- ❌ Home favori solide [1.30-1.50) : ROI -11%

### 🥇 FS (J34-J38) — "Fin de saison"
**Picks GOLD :**
1. 🔥 **HOME long shot [5+]** : ROI +55.4% ! (record absolu)
2. 🔥 Home underdog [3.50-5.00) : ROI +22.6%
3. 🔥 **Spurs home** : +23pp ! (76% WR)
4. 🔥 **C. Palace home** : +21pp ! (73% WR)
5. 🔥 **Sunderland home** : +11pp (rare upsets)
6. 🔥 HT=X → FT=1 à 44% (énorme value bet HT)

**TRAPS FS :**
- ❌ **Away favori modéré [1.50-1.80) : ROI -43.6%** ! (TRAP RECORD)
- ❌ Away léger favori [1.80-2.20) : ROI -36.7%
- ❌ Home équilibré [2.20-2.70) : ROI -30.8%
- ❌ Everton home (18% WR -17pp)
- ❌ A. Villa home (-11pp)

---

## 🔧 Module Predictor : `strategy_engine.py`

Module Python prêt à l'emploi :
```python
from scraper.strategy_engine import StrategyEngine, label_segment, print_evaluation
engine = StrategyEngine()
ev = engine.evaluate(team_a="Brighton", team_b="Manchester Red",
                      journee=35, odds_h=1.65, odds_d=4.20, odds_a=4.80)
print_evaluation(ev)
```

Output exemple :
- Détecte segment (FS si J34-J38)
- Liste signaux POSITIFS (Brighton FS +8pp, etc.)
- Liste TRAPS (cote bucket à éviter)
- Pick recommandé avec strength score
- Notes contextuelles (FS HT_X edge, base rates...)
