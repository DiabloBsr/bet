# PROPOSITION WATCHLIST v3 — définitions GELÉES (2026-06-12)

> Remplace `data/watchlist_registry.json` (v2, gelée 2026-06-11). Ce fichier est une
> PROPOSITION : aucune modification des fichiers existants. Toute définition ci-dessous
> est FROZEN : aucun paramètre (seuil, ligue, marché) ne peut être modifié après mise en
> piste — un signal modifié = un signal NOUVEAU qui repart de zéro.

## Règles du protocole v3

- **Données** : forward-only à partir du gel (2026-06-12T00:00Z) ; cote d'OUVERTURE
  = snapshot MIN(id) ; events `exports/corrupted_events.json` exclus ; dedup signature
  (même competition/team_a/team_b + mêmes cotes d'ouverture + même goals_json non-nul
  → MIN(id)) ; filtre `competition` explicite (collisions de noms) ; settlement
  FTTS/HT-FT/minutes via goals_json.
- **Critères de PROMOTION** (signal → edge misable) : `z ≥ 2` vs breakeven **ET**
  `n ≥ 80` paris forward **ET** `ROI > 0` sur la définition gelée, **ET** ROI > 0 sur le
  dernier tiers de l'échantillon (anti-drift).
- **Critères de RÉTROGRADATION / KILL** (symétriques, s'appliquent aussi aux edges en
  production) : `z ≤ −2` à tout moment, **OU** `n ≥ 300` avec ROI < 0, **OU** ROI rolling
  1000 paris < 0 pour un edge actif.
- **Multiplicité** : la v3 compte 8 slots ; toute p-value de promotion est rapportée brute
  ET avec Bonferroni ×8.
- **Un event = un ticket** : si plusieurs slots déclenchent sur le même match, prendre le
  slot de rang le plus élevé, ignorer les autres (pas d'empilement corrélé).

## Slots

### A1 — FTTS_CHAMP_HOME *(ACTIF — production surveillée, le seul edge misé)*
- **Déf gelée** : cote 1X2 domicile à l'ouverture ≤ 1,50 → back « FTTS = 1 » à la cote
  d'ouverture du même snapshot. Ligues : 8035, 8036, 8037, 8042, 8043, 8044 UNIQUEMENT
  (jamais 8056/8060/8065). 0-0 = perdu.
- **Référence** : cote moy 1,369 · WR 77,4 % · ROI corrigé +4,9 % (IC95 [+2,3 ; +7,4]) ·
  ~3 366 déclencheurs/jour. Mise flat 1 u. Kill-switch standard.
- Suivi par ligue obligatoire (8036/8042 retombés ~0 % : si une ligue passe z ≤ −2
  individuellement avec n ≥ 300, l'exclure du périmètre — c'est une rétrogradation,
  pas une modification de définition).

### A2 — FTTS_CHAMP_HOME_PREKO *(WATCHING — variante opérationnelle de A1)*
- **Déf gelée** : identique à A1, restreinte aux snapshots capturés STRICTEMENT avant
  `expected_start`. Hypothèse : +6,4 % (n=550, p=0,008 en vérification adversariale).
- But : valider que la chasse à la fenêtre de publication pré-kickoff (amélioration
  scraper) délivre le ROI supérieur. Si promue, A2 remplace A1.

### W1 — MITPS_LONGSHOT *(WATCHING — candidat cote ≥ 1,6, hérité S3/edge #5)*
- **Déf gelée** : back « 1 » au marché Mi-temps 1X2 quand la cote 1X2 home ≥ 4,0
  (sélection mi-temps cote ≥ ~11). 9 ligues, stratifié championnat/coupe.
- Référence historique : +59,8 % (n=136, 8035-old) — JAMAIS vérifié adversarialement.
  0 mise avant promotion. C'est le candidat prioritaire pour l'objectif « cote élevée ».

### W2 — FOLLOW_DRIFT *(WATCHING — hérité edge #8, nécessite re-scrape tardif)*
- **Déf gelée** : si la cote 1X2 du DERNIER snapshot pré-kickoff dévie de l'ouverture de
  ≥ 0,03 logit vers un côté → back ce côté au dernier prix. 9 ligues.
- Référence : OOS30 +11,5 % (n=197, 8035-old), non vérifié. Dépend de la capacité du
  scraper à re-snapshotter avant le round.

### W3 — VALUE_JITTER_PAIR *(WATCHING — hérité edge #7)*
- **Déf gelée** : back le côté dont la cote publiée > juste cote historique de la paire
  (n_prior ≥ 8 occurrences de la paire dans la MÊME ligue, EV estimée ≥ 0,98 hors marge).
  Mécanisme prouvé (99 % variance = jitter de publication, std 0,036 logit).
- Les nouvelles ligues accumulent l'historique de paires nécessaire ; activable par ligue
  quand n_prior médian ≥ 8.

### W4 — HTFT_X2_COMEBACK *(SUSPENDU — 0 mise, à statuer)*
- **Déf gelée** : back X/2 quand favori domicile (déf. exacte de l'acquis S1).
- Deux findings wf4 le donnent mort (HT/FT imbattable à l'ouverture ; X/2 ne survit ni au
  walk-forward ni aux nouvelles ligues) mais SANS verdict adversarial. Le slot existe
  uniquement pour produire un verdict de clôture propre : kill attendu à n ≥ 300.

### W5 — LAG1_8036_SURPRISES *(OBSERVATION pure — non bettable)*
- **Déf gelée** : autocorrélation lag-1 des surprises de round, ligue 8036 uniquement
  (r=+0,28, p=0,0016, anomalie isolée parmi 159 tests). Aucune mise possible de toute
  façon (résultats du round N publiés après le kickoff de N+1) : suivi purement
  diagnostique du moteur, verdict à n_rounds ≥ 500.

### W6 — slot LIBRE (réserve)
- Réservé au premier candidat issu du re-audit corruption 9-ligues ou de la ré-estimation
  des ratios new-era (ex : value sur cellules 2-1/1-2 si un marché à marge < 19 % apparaît).

## PURGES vs watchlist v2 (avec motif)

| Signal v2 | Motif de purge |
|---|---|
| `fade_serie_5plus`, `fade_serie_5plus_draw` | Null séquentiel définitif (LRT p=0,35 poolé-9 ; fondamentales = zéro info, 0/22 claims) |
| `sous_regime_rebond` | Idem — le sur/sous-régime ne prédit rien ; le +40 % d'origine = artefact petit-n |
| `standings_pos_gap5`, `standings_pts_gap5` | Classement = zéro info au-delà des cotes (acquis re-confirmé) |
| `value_home_vs_alltime` | Famille fondamentales ; en plus mal posé multi-ligues (collisions de noms) |
| E2 / favori extrême [1,10-1,20] | MORT : −7,57 %, calibré parfait, p=1,7e-6 vs breakeven |
| « Total de buts = 1 » (S2 / edge #3) | MORT : −8,2 %, 9/9 ligues négatives |
| Règles segment DS/MS (P1-P5) | Artefacts full-sample, retraits confirmés |
| `value_jitter_pair`, `mitps_longshot_global`, `follow_drift` | NON purgés — repris en W3/W1/W2 avec définitions gelées |

## Interdictions permanentes (hors watchlist — ne jamais re-tester sans mécanisme nouveau)

Multi-Buts (marge 98 %) · cotes 100 (cellules cappées) · payoffs dépendant de 7+ buts ·
O/U 3,5 systématique (les deux côtés) · G/NG toutes zones, surtout Non avec |λh−λa| ∈
[0,6 ; 1,3) · combos 1X2&G/NG et 1X2&Total · totaux exacts · 1X2 mid-odds 1,8-5,0 sur la
seule cote · FTTS en coupes · conditionnels intra-round et round N→N+1.
