# Bulletin de santé du système — Audit WF5 du 2026-06-12

**Périmètre** : InstantLeague-8035, 2026-05-28 → 2026-06-12, ~7 391 matchs propres avec cotes d'ouverture (MIN(id) par event), 388-473 events corrompus exclus selon les fenêtres.
**Méthode** : 5 audits indépendants (stationnarité cœur, signaux de paires, segments/buckets, watchlist forward, scores V2) + contre-vérification adversariale des 12 verdicts critiques. **12/12 verdicts critiques CONFIRMÉS, 0 renversé** (confiance 0.85–0.93).

**Caveats transverses** :
- Couverture cotes : seuls **49 %** des matchs finis ont un snapshot (32 % aujourd'hui) — le scraping ne suit plus depuis l'ajout des 8 ligues cette nuit. Conclusions limitées aux matchs scrapés pré-kickoff.
- Snapshots d'ouverture capturés en médiane +39 s après le kickoff nominal (cotes virtuelles figées : pas de fuite, mais à scraper avant KO en prod).
- ~3 130 matchs finis du 10-12/06 ont des lignes events/results dupliquées (artefact scraping multi-ligues) — impact vérifié négligeable sur tous les verdicts.

---

## 1. Tableau signal par signal

### 1.1 Cœur du moteur & edges principaux

| Signal | Statut | n | ROI | Preuves clés | Action |
|---|---|---|---|---|---|
| **E1 — FTTS « 1 » favori home ≤1.50** | ✅ **SAIN** | 1 714 | **+6.66 %** (z=4.73) | Mécanisme ratio réel/implicite 1.193 intact ; +4.67 % sur le seul 06-12 (n=452) ; CUSUM p=0.247, pas de rupture | **GARDER**, sizing 0.8u inchangé. Alerte si ROI glissant <+2 % sur 500 prochains picks |
| **E2 — 1X2 favori extrême [1.10-1.20]** | ❌ **OVERFIT** (contre-vérif ✔ 0.93) | 365 | **-0.84 %** (z=-0.39) | Jamais rentable après marge : edge calibration +4.2pp < marge 6 % ; le « +5.45 % OOS » = fenêtre chanceuse 04-05/06 (WR 95.6 %, n=45) ; répliqué **négatif sur les 8 autres ligues** (pool n=660, -7.31 %) | **DÉMOTER : stake 0**. Re-promotion seulement si un filtre additionnel porte le WR >86 % (break-even) |
| **TIER 1 gate (proxy favori ≤1.30)** | ❓ **INCONNU** | 849 | -3.23 % | Gate V5 non reproductible rétroactivement ; baseline WR 79.7 % stable vs 82 % annoncé ULTRA — tout repose sur la valeur ajoutée V5, invérifiable | **Audit dédié requis** : logger les picks TIER1 réels avec proba V5. Ne pas augmenter le sizing en attendant |
| **MOTEUR — régime des favoris (CUSUM ≤1.50)** | ✅ **SAIN** | 2 161 | -4.44 % (baseline) | CUSUM p=0.90 : aucune rupture ; boost favori +1pp constant ; seul artefact = cold streak transitoire soirée du 06-05, résorbé le 06-06 | Conserver ENGINE_MODEL tel quel. Ajouter un CUSUM hebdo automatisé (seuil alerte p<0.01) |

### 1.2 Signaux de paires (team_gold_data.py) — overfitting quasi-total confirmé

Leçon systémique : les μ de paire sont fixes ET déjà pricés à 6 % de marge — toute stat de paire sur n=8-12 au-delà des cotes est du bruit.

| Signal | Statut | n OOS | ROI OOS | Claim → réel | Action |
|---|---|---|---|---|---|
| PAIR_HOME_GOLD | ❌ MORT (✔ 0.92) | 171 | -6.8 % | WR 81 % → 50.9 % (= marché) | Retirer du predictor_v10 (boost « ++ » +0.25 EV = bruit pur) |
| PAIR_AWAY_GOLD | ❌ MORT (✔ 0.93) | 119 | +1.7 % (z=0.12) | WR 58.6 % → 34.5 % (= implicite 33.4 %) | Retirer (claim +50/+120 % = artefact 8-12 tirages) |
| PAIR_TRAP_HOME | ❌ MORT (✔ 0.93) | 37 | — | « home 0 % » → WR home OOS 59.5 % = moyenne ; pire trap gagne 8/9 | **Retirer D'URGENCE le block absolu (-1.0)** : bloque des paris valides |
| BRACKET_GOLD_HOME | ❌ MORT (✔ 0.90) | 463 | -4.8 % | Converge vers -marge ; 6 variantes de sensibilité toutes négatives | Retirer du predictor_v10 |
| BRACKET_GOLD_AWAY | ❓ INCONNU | 625 | +8.9 % (z=0.99) | Seul bracket positif, jamais significatif | Ne pas miser ; watchlist forward (z≥2 & n≥80 & ROI>0) |
| BRACKET_TRAP_HOME | ❌ MORT (✔ 0.85) | 409 | -2.0 % | Claims -22/-43 % rejetés à z=+3.1 ; indiscernable des sélections normales | Retirer le malus |
| OVER_GOLD | ❌ OVERFIT (✔ 0.93) | 159 | — | Hit 90 % → 62.3 % = baseline exact (shrinkage 100 %) ; **ligne O/U 2.5 jamais cotée** (que 3.5) | Supprimer la table — aucun signal, aucun instrument |
| UNDER_GOLD | ❌ OVERFIT (✔ 0.88) | 78 | — | Hit 76 % → 47.4 % vs base 37.6 % : résidu réel MAIS l'instrument réel (Multi-Buts 0-1-2) le price déjà → ROI -2.9 % | Ne pas miser ; résidu utilisable comme feature score_predictor seulement |
| BTTS_OUI_GOLD | ❌ MORT (✔ 0.93) | 117 | -11.8 % (z=-1.65) | Hit 88 % → 57.3 % = baseline ; le filtre min_cote_h AGGRAVE | Supprimer table + filtre (pire que le hasard) |
| BTTS_NON_GOLD | ⚠️ EN_DÉRIVE | 109 | +13.9 % (z=1.51) | Seul vrai résidu de paire : hit 59.6 % vs base 41.9 % (z=3.7) — Dixon-Coles non pricé | Ne pas promouvoir ; **ajouter comme 10e signal watchlist** (claims recalés : hit ~60 %, ROI ~+10 %) |
| SCORE_DOMINANT/SWEET | ❌ OVERFIT (✔ 0.90) | 200 | +33.2 % (z=1.28) | Hit 43 % → 13.5 % ; bat légèrement l'implicite (10.7 %) mais jamais significatif sur 2 fenêtres | Recalibrer les rates à ~0.13 partout ; mise interdite ; watchlist (il faut ~3× plus de n) |
| SCORE_COMBO_GOLD | ❌ MORT (✔ 0.92) | 126 | -2.9 % | Hit 61.6 % → 19.1 %, SOUS l'implicite 20.8 % ; le 2e score n'ajoute que de la marge (18 %) | Supprimer ; le résidu éventuel est dans le top1 seul |

### 1.3 COTE_EDGES — buckets cote × segment saison (strategy_engine.py)

| Signal | Statut | Preuves clés | Action |
|---|---|---|---|
| **Couche COTE_EDGES complète (~80 cellules)** | ❌ **OVERFIT** (✔ 0.93) | Corrélation ROI orig↔récent **négative** (Pearson -0.50) ; 4/5 cellules significatives inversent leur signe ; 0 réplication sur 14 cellules n≥100 ; mécanisme contredit par ENGINE_MODEL (probas paires fixes, marge plate 6 %) | **Neutraliser (poids 0)** dans la consolidation — ne PAS recalibrer, le canal causal n'existe pas |
| Portefeuille EDGES+ (11 cellules) | ❌ MORT | orig +8.8 % (n=1 384) → récent **-1.1 %** (n=661) | Cesser de miser ces buckets |
| Portefeuille TRAPS (14 skip-rules) | ❌ MORT | orig -15.6 % → récent **+3.4 %**, MIEUX que la baseline (-3.2 %) : éviter les traps a coûté de l'EV | Retirer les skip-rules (pénalité strength×0.5 dans evaluate()) |
| Trap FS away 1.5-1.8 (ex « -44 % ») | ❌ OVERFIT | récent +16.9 % (signe inversé), winner's curse classique | Supprimer |
| Trap DS home 1.8-2.2 (ex « -25 % ») | ❌ MORT | récent +23.7 % (z=+1.7) — inversion complète | Supprimer |
| Edge DS home 2.2-2.7 (ex « +20 % ») | ❌ OVERFIT | Ne se reproduit même pas en orig (+7.2 %, z=0.5) | Supprimer |
| Edge MS_early home 5+ (ex « +44 % ») | ❌ MORT | récent **-53.7 %** (z=-2.0) — effondrement total | Supprimer immédiatement |
| Edge MS_mid home 5+ | ❓ INCONNU | Signe stable mais z<1 des 2 côtés | Watchlist si on veut trancher, pas de mise |
| Edge MS_mid away 5+ (seul n_récent≥100) | ⬇️ DÉMOTER | +7.0 % récent mais z=0.4 ; « +19.6 % » jamais observé | Retirer de la production |
| **Trap MS_early away 1.8-2.2 — SEUL SURVIVANT** | ⚠️ EN_DÉRIVE | Significatif dans les 2 périodes (orig -25.8 % z=-2.7 ; récent -34.1 % z=-2.35) mais proba ~3 % par hasard sur 71 cellules, n=41<80 | Watchlist forward comme avoid-rule ; NE PAS réactiver avant confirmation |
| Dimension « segment saison » (DS/MS/FS) | ❌ MORT | Zéro feedback saison sur les probas (99 % de la variance des cotes = jitter de publication) | Abandonner la segmentation saison pour le 1X2 |

### 1.4 Watchlist forward (gels v1 11/06 17:00, v2 12/06 01:30)

Bilan : **0 PROMOUVOIR, 0 DÉMOTER, 8 CONTINUER, 1 INVÉRIFIABLE**. Aucun signal n'atteint z≥2.

| Signal | Statut fwd | n fwd | ROI fwd | z | Lecture |
|---|---|---|---|---|---|
| standings_pos_gap5 | ✅ CONTINUER | 268 | +18.7 % | +1.22 | Le plus prometteur en volume ; re-auditer dans 2-3 jours |
| standings_pts_gap5 | ✅ CONTINUER | 212 | +11.5 % | +0.88 | Recouvre probablement pos_gap5 → n'en promouvoir qu'UN |
| mitps_longshot_global | ✅ CONTINUER | 233 | +8.9 % | +0.72 | Positif mais à ~moitié du ROI histo (+18.3 %) ; ~1 500-2 000 paris requis pour trancher |
| sous_regime_rebond | ✅ CONTINUER | 67 | +4.7 % | +1.10 | n=80 sous 48h, promotion improbable au 1er palier |
| fade_serie_5plus_draw | ✅ CONTINUER | 35 | +33.1 % | +1.40 | Compatible bruit, ne pas miser |
| value_home_vs_alltime | ✅ CONTINUER | 25 | +20.1 % | +0.81 | Très faible débit, ~4-5 jours avant verdict |
| fade_serie_5plus | ⚠️ EN_DÉRIVE | 35 | -3.4 % | +0.18 | Négatif histo ET forward → démotion probable à n≥80 |
| **value_jitter_pair** | ❌ **OVERFIT de fait** | **1 477** | **-0.5 %** | +1.21 | Sur 4× l'OOS d'origine, le « +18.6 % » s'effondre à ~0 : estimation d'origine contaminée. **NE PAS MISER** (pas démotable par la règle). Option : geler une variante EV≥1.02-1.05 (nouveau signal) |
| **follow_drift** | ❓ INVÉRIFIABLE | **0** | — | — | 0 pari possible : 1.25 snapshot/event post-gel (vs 3.06). Rétablir un re-scrape tardif pré-KO ou retirer le signal |

### 1.5 Score Predictor V2 vs V5

| Signal | Statut | Preuves clés | Action |
|---|---|---|---|
| **V2 — claim « +9pp Top1 / +15pp Top3 »** | ❌ **OVERFIT (fuite de données)** | Replay leak-free (n=2 410) : V2 fait **-0.41pp Top1 / -0.75pp Top3** vs V5 ; en réinjectant le pair cache qui lit toute la BDD (résultats de test inclus) on retrouve +12.8/+17.9pp = le claim. C'est une fuite, pas une capacité prédictive | Retirer les mentions « +9pp/+15pp » de scripts/_predict_one_round.py (lignes 51 et 134) ; ne pas promouvoir V2 |
| **Probas affichées V2 (ex : « 2-1 à 39 % »)** | ❌ MORT | Tout annoncé ≥20 % retombe à ~11-13 % réel (gap -11 à -20pp). Cause : sources non normalisées (PROFILE top3 masse 0.29-0.59) + renormalisation sur union tronquée + pair n≥5 bruité | **Ne JAMAIS sizer une mise sur la proba V2** (un « 39 % » vaut ~10-12 %) |
| V5 brut — score exact | ✅ SAIN | Top1 11.99 %, Top3 29.71 % = backtest historique, proche du plafond théorique 13-14 % ; proba Top1 bien calibrée (annoncé 11.4 % → réel 11.9 %) | Garder V5 (top5_scores_enriched) comme seul prédicteur de score affiché |
| Calibration 1X2 V5 (gate TIER1) | ✅ SAIN | Annoncé 70-75 % → WR 75.2 % ; 75-80 % → 83.0 % ; légèrement sous-confiant en haut (Brier 0.2345) | Conserver le gate tel quel ; surveiller le bucket 70-75 % de p_cote (-4.4pp, dans l'IC) |

---

## 2. Performance live du jour (2026-06-12, 06:00–16:45 UTC)

1 780 matchs analysés sur 5 560 finis (**32 % de couverture cotes seulement** — la cadence scraper ne suit pas les rounds).

| Famille | n jour | WR jour | ROI jour | vs historique | Verdict du jour |
|---|---|---|---|---|---|
| **E1 FTTS favori home ≤1.50** | 414 | 78.5 % | **+8.7 % (z=+3.09)** | Histo +7.5 % | 🏆 **Gagnant du jour** : seule famille rentable ET significative, en ligne avec l'histo. Le socle du portefeuille |
| TIER1-approx (favori ≤1.30) | 194 | 79.9 % | -2.7 % | Histo 79.7 % / -3.4 % | Flux de probas stable (zéro dérive moteur) ; le proxy brut paie la marge — seul le gate calibré peut être positif |
| E2 favori extrême [1.10-1.20] | 82 | 80.5 % | -6.3 % | Attendu 86.6 % | Jour à -1.6σ ; cohérent avec le verdict OVERFIT — démoter |
| BTTS NON si home ≤1.30 (proxy) | 169 | 47.9 % | **-11.9 %** | Histo -5.1 % | ❌ Négatif sur les DEUX périodes : retirer immédiatement de la rotation |
| COMBO scores (SCORE_COMBO_GOLD) | 73 | hit 20.5 % | +0.6 % (z=0.02) | Annoncé 62 % (z=-7.4 !) | Overfit démontré en une seule journée OOS |
| SWEET score dominant | 117 | hit 15.4 % | +58.4 % (z=+1.5) | Annoncé 43.5 % (z=-6.2) | Hit = plafond du top1 générique (13-14 %) : les paires n'apportent RIEN ; le +58 % = 2 hits à cote ~21.7, pur bruit — ne pas augmenter la mise |

---

## 3. Actions concrètes priorisées (PROPOSITIONS — aucun fichier modifié par cet audit)

### P0 — Immédiat (arrêter de perdre / de bloquer de l'EV)
1. **predictor_v10.py : retirer le block absolu PAIR_TRAP_HOME** (boost -1.0, lignes ~333-334) — il bloque activement des paris valides (le pire « trap » gagne 8/9 OOS).
2. **E2 → stake 0** (exotic_signals.py) ou retour watchlist avec critère strict « WR >86 % avec filtre additionnel ». Jamais rentable après marge ; répliqué négatif sur les 8 autres ligues.
3. **Retirer le proxy BTTS NON home_crush de la rotation** : -11.9 % aujourd'hui, -5.1 % histo — jamais rentable.
4. **strategy_engine.py : neutraliser la couche COTE_EDGES (poids 0)** — les edges+ déclarés ET les skip-rules traps (la pénalité strength×0.5 dans evaluate() coûte de l'EV). Ne pas recalibrer.
5. **Ne JAMAIS dimensionner une mise sur la proba affichée V2** (un « 39 % » vaut ~10-12 % réels).

### P1 — Cette semaine (nettoyage des tables mortes)
6. **predictor_v10.py : retirer les signaux PAIR_HOME_GOLD, PAIR_AWAY_GOLD, BRACKET_GOLD_HOME, BRACKET_TRAP_HOME** (boosts ++/-- et ±EV = bruit pur).
7. **team_gold_data.py : supprimer OVER_GOLD** (aucun instrument : la ligne 2.5 n'est jamais cotée), **BTTS_OUI_GOLD + son filtre min_cote_h** (le filtre aggrave), **SCORE_COMBO_GOLD**.
8. **Recalibrer les rates SCORE_SWEET/DOMINANT à ~0.13** (pas 0.4-0.6) partout où la table est consommée ; mise interdite.
9. **scripts/_predict_one_round.py : retirer les mentions « +9pp/+15pp » et « Top1 22 % vs 12 % »** (lignes 51 et 134) — artefact de fuite de données ; afficher V5 brut seul.

### P2 — Instrumentation (sans quoi les prochains audits seront aveugles)
10. **Logger les picks TIER1 réels avec la proba V5 au moment du pick** (forward) — c'est le seul moyen de vérifier les WR 82/78/72 % annoncés. Ne pas augmenter le sizing avant.
11. **Rétablir un re-scrape tardif pré-kickoff sur 8035** (2e snapshot par round) : sans lui, follow_drift est invérifiable (0 pari possible) et la couverture cotes s'effondre.
12. **Capacité scraping : prioriser 8035 pré-KO sur les 8 nouvelles ligues** — couverture tombée à 32-49 %, ce qui ralentit ~3× l'accumulation de preuve de TOUTE la watchlist.
13. **Automatiser le CUSUM hebdomadaire** (réutiliser scripts/_wf5_stationarity_audit.py tel quel) comme canari de régime, alerte à p<0.01.
14. **Mettre à jour exports/corrupted_events.json pour les events du jour** + dédupliquer les events multi-lignes du 10-12/06.

### P3 — Watchlist (gels et suivis, définitions figées)
15. **Ajouter 3 entrées** : BTTS_NON_GOLD (claims recalés hit ~60 %, ROI ~+10 %), BRACKET_GOLD_AWAY, trap MS_early away 1.8-2.2 (en avoid-rule). Gates standard : promotion z≥2 & n≥80 & ROI>0.
16. **value_jitter_pair : ne pas miser** (édge d'origine contaminé, -0.5 % sur n=1 477) ; si on veut le sauver, geler une NOUVELLE variante seuil EV≥1.02-1.05 sans toucher à la définition figée.
17. **standings_pos_gap5 / pts_gap5 : re-auditer dans 2-3 jours** ; forte intersection probable → au moment d'une promotion, en choisir UN seul.
18. **E1 : règle de surveillance** — si ROI glissant <+2 % sur les 500 prochains paris E1, repasser en watchlist.

### Ce qui reste misable aujourd'hui
| Famille | Sizing proposé |
|---|---|
| **E1 FTTS favori home ≤1.50** | 0.8u (inchangé) — seul signal rentable, significatif et confirmé live |
| TIER1 via gate calibré uniquement | inchangé, en attente de l'audit forward des picks réels |
| Tout le reste (E2, paires GOLD, COTE_EDGES, SWEET/COMBO, watchlist) | **0u** — watchlist ou suppression |

---

## 4. Annexes — reproductibilité

| Audit | Scripts | Résultats |
|---|---|---|
| Stationnarité cœur | scripts/_wf5_stationarity_audit.py, _wf5_stationarity_checks.py | exports/wf5_stationarity.json, wf5_stationarity_checks.json |
| Paires GOLD | scripts/_wf5_pair_gold_audit.py (+ rechecks _wf5_pair_gold_recheck.py, _wf5_pair_away_gold_recheck.py, _wf5_pair_trap_recheck.py, _wf5_bracket_gold_recheck.py, _wf5_trap_bracket_recheck.py, _wf5_over25_market_check.py, _wf5_under_gold_counterverify.py, _wf5_btts_oui_counterverify.py, _wf5_combo_counterverify.py, _wf5_pair_gold_dedup_recheck.py) | exports/wf5_pair_gold_audit.json + exports de contre-vérification |
| Buckets/segments | scripts/_wf5_bucket_audit.py, _wf5_bucket_audit2.py, _wf5_bucket_audit_verify.py | exports/wf5_bucket_audit.json, wf5_bucket_audit2.json, wf5_verify.json |
| Watchlist | définitions gelées de scripts/_signal_watchlist.py (répliquées lecture seule) | exports/wf5_watchlist_audit.json |
| Scores V2 | scripts/_wf5_score_v2_audit.py, _wf5_score_v2_leak_check.py | exports/wf5_score_v2_audit.json, wf5_score_v2_leak_check.json |
| E2 contre-vérif | scripts/_wf5_e2_counterverify.py | exports/wf5_e2_counterverify.json |

*Rapport généré le 2026-06-12 — audit en lecture seule, aucun fichier de production modifié.*
