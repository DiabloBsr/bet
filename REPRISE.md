# Reprise de session — Bet261

> Note de passation écrite le 04/10/2026. Fichier NON suivi par git :
> à committer ou à supprimer selon ce que tu préfères.

## 1. Coordonnées du chantier

| | |
|---|---|
| Dépôt local | `d:/projets/bet` |
| Branche | `feat/h2h-toutes-ligues` |
| Distant | `DiabloBsr/bet` — **jamais** l'org Agentova |
| Identité git | `DiabloBsr` / `diablobsr@users.noreply.github.com` |
| Déploiement | Hugging Face `olivioBsr/vfoot-trio` |
| Commande | `.venv/Scripts/python.exe deploy/deploy_hf.py` |
| Python | `.venv/Scripts/python.exe` (Windows) |
| Tests | `.venv/Scripts/python.exe -m pytest tests/ -q` |

## 2. ÉTAT EXACT À LA REPRISE

| Point | État |
|---|---|
| Dernier commit | `287de6c` — prédiction du round en tableau + over/under 2,5 |
| Sport réel (Elo, ESPN) — 4 sports | **FOOT DÉPLOYÉ + NFL/NBA/NHL en local** (20/10). Projet « prédiction sport RÉEL ». Données **API publique ESPN** (gratuit, sans clé) : `site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates=YYYYMMDD` (jour par jour, pas de plage → fetch **threadé**). `sport_reel.py` : fetch/parse + **Elo** (avantage terrain par sport, nul au foot) → probas. Onglet « ⚽ Sport réel ». `SPORTS` = 6 ligues foot (nul, home_adv 60) + **NFL** (football/nfl, 48) + **NBA** (basketball/nba, 100) + **NHL** (hockey/nhl, 50), sans nul. Foot **vérifié en ligne** (HF, Arsenal 62/13/25 vs Everton…). NFL/NBA/NHL smoke local OK (Bills, Knicks, Hurricanes en tête). 8 tests, 535 au total. **Ne bat PAS le marché.** ⚠️ **Tennis de table : PAS sur ESPN** (400) → non livré. À la place, **Tennis ATP + WTA ajouté** (Olivio : « ça m'intéresse ») : structure ESPN différente (event=tournoi → `groupings`[tableaux] → `competitions`[matchs], joueurs via `athlete`, résultat via flag `winner` car pas de score chiffré). `kind="tennis"`, `home_adv=0`, filtre `slug` (mens-singles / womens-singles) car les flux atp/wta renvoient le MÊME tournoi avec tous les tableaux (sinon on mélange H/F + doubles). `SPORTS` passé en dict-de-dict. 537 tests. Smoke local OK (ATP top Zverev/Sinner, WTA top Gauff/Swiatek). NFL/NBA déployés (HF) ; tennis déployé (HF, ATP top Sinner/Zverev vérifié en ligne). |
| Sport réel — pronostic forme + face-à-face | **FAIT (local, NON poussé)** (20/10). Olivio : « pronostics qui réussit, pas sur les cotes mais sur forme + confrontation ». `sport_reel.py` : `forme` (6 derniers, V/N/D + taux), `face_a_face` (h2h direct), `pronostic_forme` (part de forme + bonus terrain + h2h si ≥2 confrontations, nul au foot), `pronostics_a_venir`, `taux_reussite` (backtest honnête walk-forward). Onglet refondu : pronostic (pick+confiance) + forme + face-à-face + **taux de réussite affiché**. 4 tests, 541 au total. **Backtest mesuré : NBA 64% (base dom 58%), tennis ATP 56% (base 46%) → la forme+h2h RÉUSSIT ; foot ~40% (même l'Elo 42%, 1X2 intrinsèquement dur).** AUCUNE cote utilisée. À DÉPLOYER sur GO. |
| Sport réel — Top 3 du jour par sport | **FAIT (local, NON poussé)** (20/10). Olivio : « 3 meilleures prédictions de chaque sport tous les jours ». Nouvelle section « 🏆 Top 3 du jour » : pour chaque sport (Football=6 ligues, NFL, NBA, NHL, Tennis=ATP+WTA), les 3 pronostics forme+face-à-face les plus CONFIANTS sur les matchs des 3 prochains jours. `sport_reel.groupe_de`/`GROUPES`. Loader léger caché 3h (hist 150j, fx 3j, sans backtest). « Tous les jours » = la vue affiche toujours les matchs à venir, cache 3h. Smoke NBA OK (Clippers 97%…). 541 tests. ⚠️ Premier calcul lent (~1-2 min, charge 10 flux ESPN) puis caché. Version AUTOMATIQUE pré-calculée (sans attente / notif) possible en Étape suivante si demandé. À DÉPLOYER sur GO. |
| Sport réel — blessures | **FAIT (local, NON poussé)** (20/10). Olivio : « prendre en compte les joueurs blessés ». `sport_reel.blessures_du_sport(path)` lit l'endpoint ESPN `/injuries` → {équipe: [{nom,status}]} (Out / Day-To-Day). **Dispo NBA/NFL/NHL ; VIDE au foot (ESPN ne publie pas) et tennis (forfait=pas de match).** `poids_blessures` (Out=1,0 ; DTD=0,4). `pronostic_forme` décale la proba contre l'équipe la plus touchée (0,04/unité, cap ±0,12) et renvoie bless_home/away. Onglet : colonne « Absents (dom/ext) » ; Top 3 : noms des absents. ⚠️ COMPTE pondéré, pas de valeur par joueur (perdre une star = perdre un remplaçant) → repère, pas mesure exacte ; pas backtestable (blessures = instantané). 543 tests. Smoke NBA OK (Gueye/Veesaar Out ; Detroit-Boston Carter Out). DÉPLOYÉ (HF, colonne Absents vérifiée en ligne, Knicks 89% vs Philly 0/3). |
| Top 3 — date, championnat, heure, analyse | **FAIT (local, NON poussé)** (20/10). Olivio : dans le Top 3, ajouter date/heure + championnat de chaque match, mon propre pronostic + analyse, et un champ de date au choix. `sport_reel` : `pronostics_a_venir` porte la `date` ; `matchs_du_jour(path, jour)` (fenêtre 1 jour) ; `analyse_match(p)` (texte FR : forme + face-à-face + absents + « 👉 Mon pronostic »). Onglet Top 3 : `st.date_input` (défaut aujourd'hui Mada), chaque pick montre championnat + date/heure Mada + analyse. 546 tests. Smoke NBA OK (Knicks 81%, analyse citant les 3 absents de Philly). DÉPLOYÉ (HF, Top 3 vérifié en ligne : Fenerbahce 78%, championnat+heure+analyse OK). |
| Sport réel — marchés élargis | **FAIT (local, NON poussé)** (20/10). Olivio : « pas que le vainqueur » → les deux marquent, total buts/corners (foot), total points + meilleur marqueur (NBA/NHL). TOUS sans cote, depuis les scores/leaders ESPN. (1) **Totaux + BTTS** : `stats_off`, `marches_buts` (foot = BTTS + plus/moins 2,5 via Poisson ; NBA/NHL = total vs ligne=moyenne ligue). (2) **Meilleur marqueur** NBA/NHL : `_leader_points` (dans le scoreboard, SANS fetch en plus) + `meilleur_marqueur`. Foot : pas de buteurs exposés par ESPN. (3) **Corners** foot : `stats_corners`/`marche_corners` via endpoint `summary` (1 requête/match → opt-in case à cocher dans l'onglet détaillé, PAS dans le Top 3). `analyse_match` + colonne « Marchés » affichent tout. 553 tests. Smoke OK (EPL BTTS non + moins 2,5 ; NBA Wembanyama ~26 ; NHL +6 ; corners Chelsea 6,8). |
| Sport réel — tennis total de sets | **FAIT (local, NON poussé)** (20/10). Olivio : « pour le tennis total de set ». Faisable : le match tennis porte `linescores` (set par set) → `n_sets = len(linescores)` ET `format.regulation.periods` (best-of), **déjà dans le scoreboard, sans fetch**. `_parse_tennis` capte n_sets+periods ; `marche_sets` (moyenne sets récents des 2 joueurs → total attendu, ligne 2,5 en bo3 / 3,5 en bo5). `pronostics_a_venir(kind=…)` : tennis reçoit `sets` (pas les marchés de buts). `analyse_match` + colonne Marchés affichent les sets. 555 tests. Smoke ATP OK (1952 matchs, moy 2,66 sets ; Hijikata-Atmane « moins de 3,5 »). DÉPLOYÉ (HF, colonne Marchés tennis vérifiée en ligne). |
| Sport réel — tennis total de jeux | **FAIT (local, NON poussé)** (20/10). Olivio : « ajoute aussi le total de jeu ». `linescores.value` = jeux par set → `n_games` = somme des jeux des 2 joueurs (déjà dans le scoreboard). `_parse_tennis` capte n_games ; `marche_jeux` (ligne 22,5 bo3 / 36,5 bo5). `pr["jeux"]` ajouté au tennis ; analyse + colonne Marchés. 556 tests. Smoke ATP OK (moy 26,3 jeux ; « moins de 36,5, ~25 attendus »). À DÉPLOYER sur GO. |
| Module Courses — Palier 1 | **FAIT + DÉPLOYÉ** (19/10) — HF `2871dee` puis fix `b201dfc` (urllib au lieu de httpx, absent de l'image — faisait tomber TOUTE l'app). Onglet « 🏇 Courses » (chevaux 8006 + lévriers 8005) + `courses_bet261.py` (fetch, dévig, favori). Vérifié en ligne, app stable. 6 tests |
| Module Courses — Palier 2 Étape A (collecte) | **FAIT (local, NON poussé)** (20/10). RÉSULTATS récupérables : re-GET `events/{id}` après le départ porte `result` (ordre d'arrivée) + `runners` (noms) une fois `Settled` (mon diagnostic « bloqué » du 19/10 était FAUX — corrigé dans [[racing-api-bet261]]). `courses_collect.py` (table `courses`, upsert cotes + collecte arrivées), `_courses_loop.py` (boucle urllib+sqlite, `start_cloud.sh` --interval 60, log /data/courses.log). 5 tests, +DEFAULT_FILES. 527 tests. **Prouvé live** : +42 captées puis 1 réglée (Milnerton, arrivée [4,10,6,3,…], noms). ⚠️ /data repart de zéro au rebuild (pas de seed courses). **Étape B (analyse biais RNG + proba indépendante) à faire une fois des données accumulées.** À DÉPLOYER sur GO d'Olivio |
| Entraînement par ligue | **FAIT** (11/10) — `19be676`, commit HF `5077ce4`. V2/V5 s'entraînent sur CHAQUE ligue. Vérifié en ligne sur l'Allemagne : « Moteurs O/U » à 3/3 |
| Cibles 🔥🧊⚡ | **FAIT** (11/10) — `26f95a8`, HF `63f46d2`. Tableau du round : 🔥 tête du classement, 🧊 meilleure attaque, ⚡ meilleure forme (5 derniers). `cibles_saison`→{feu,glace,eclair}. Vérifié en ligne (London Reds 🔥🧊⚡) |
| Que jouer épuré + Débusqueur retiré | **FAIT** (15/10) — `190260b`, HF `f658d47`. « Que jouer ? » : filtres « Cote min. » et « Conseil retenu » retirés (plafond « Rencontres max » gardé — mesure en ligne : 1080 rencontres à venir, d'où le plafond vital) ; détail réduit à Score exact / 1X2 / G/NG. Onglet Débusqueur effacé (moteur `debusquer_rounds`/`round_1x2` conservé). Reste 2 onglets. 494 tests |
| Round épuré (colonnes retirées) | **FAIT + DÉPLOYÉ** (16/10) — `90bb8f5` (combiné de 3 matchs retiré de l'écran, `combine_round` conservé non câblé) + `a4bc834` (colonnes « Moteurs O/U » et « Classement » retirées de l'affichage ; le rang passe à côté du nom des équipes sous le tableau via `_nom_rang`/`_pt._rang`). `_cols_round = (Top 3, Match, Pronostic, Cote, Score exact, Sinon)`. Poussé `origin` + HF `260370b`, app stable 5/5 en 200. Vérifié en ligne (CAN round 22:55) : 6 colonnes seules, « South Africa (4e) / Gabon (24e) » et « Zambia (17e) / Ivory Coast (1er) » sous le tableau. 507 tests |
| Emoji 2-1/1-2 CAN | **FAIT + DÉPLOYÉ** (19/10) — `6deaf60` (⚽ sans condition) puis `2e80dc0` (HF `40c6dbf`). Un 2-1/1-2 en CAN suit la MÊME condition que le 1-1 (🔒💀🔻🔺, via `emoji_nul_11` sur le dernier match de chaque équipe) ; si aucune condition ne s'applique → ⚽. `_marque_21(sc, m, derniers)` + `_marque_score`, sur score principal ET « Sinon ». Légende dashboard (2 lignes), tests `test_marque_21.py` (9) + `test_nul_11` adapté. 516 tests. Vérifié en ligne (19/10) : légende OK, 1-1 annoté (Nigeria-Zambia « 1-1 15 % 🔺 ») ; ⚠️ aucun 2-1/1-2 dans le round testé pour montrer ⚽ en direct — à reconfirmer. App stable 5/5 en 200 |
| Scraper cible 40s avant coup d'envoi | **FAIT + DÉPLOYÉ** (19/10) — `4cd43cc` (dans push `6deaf60`). `start_cloud.sh` : intervalle 40s → **30s**. Cible de préparation relevée à 40s avant le coup d'envoi ; marge pire cas ≈ 80s − intervalle, donc 30s → ~50s (coussin), 40s n'aurait donné que ~40s (limite). App stable |
| Scraper 40s | **FAIT puis REMPLACÉ** (18/10) — `78fdd74`, HF `3173275`. `start_cloud.sh` : `--interval 30` → **40** (Olivio : « 40, ou 45 même » → 40 retenu pour garder la marge). Garantie « prêt 30s avant le coup d'envoi » tenue : rounds toutes les 120s, marge ~40-50s à 40s (était ~50-60s à 30s), tout en allégeant la charge API. Rebuild = conteneur redémarré → `start_cloud.sh` relancé avec interval 40. `/data/scrape.log` non exposé hors du Space, pas de vérif directe du log. App stable 5/5 en 200. 507 tests |
| Colonnes Score exact / Sinon élargies | **FAIT + DÉPLOYÉ** (16/10) — `83326cf` puis `bb296e0` (large → **medium**, HF `b691e5`). `_tableau_vert` : `column_config` `width="medium"` sur « Score exact » et « Sinon » quand présentes. `large` débordait sur ordinateur (Olivio) ; `medium` laisse la place aux emojis de l'annotation 1-1 (dont le double 🔻🔻/🔺🔺) en portrait sans élargir à l'excès. Vérifié en vue mobile 375px (CAN round) : contenu complet, pas de troncature interne ; la colonne « Match » épinglée reste large (noms longs), le reste se consulte en défilant (inhérent à st.dataframe sur écran étroit). ⚠️ pas de 1-1 principal avec emoji dans le round testé pour montrer le double en direct — à reconfirmer sur un round qui en a un. App stable 5/5 en 200. 507 tests |
| Mi-temps retirées de « Que jouer ? » | **FAIT + DÉPLOYÉ** (16/10) — `e14f64c`, HF `67c9989`. Affichage des pronostics « 1re mi-temps » / « 2e mi-temps » retiré de l'onglet, ainsi que la légende et la mention d'intro. Le moteur calcule toujours `res_c["mi_temps"]` (`marches_mi_temps`, calibration intacte, le pari suggéré reste le meilleur des 11 marchés). Test `test_mi_temps.py::test_l_onglet_n_affiche_plus_les_deux_periodes` adapté. App stable 5/5 en 200. Vérifié en ligne (UCL round 23:38 : détail = G/NG · 1X2 · Score exact, aucune mi-temps). 507 tests |
| Pronostic en fin de tableau | **FAIT + DÉPLOYÉ** (16/10) — `121bbad`, HF `91c0b2d`. `_cols_round = (Top 3, Match, Cote, Score exact, Sinon, Pronostic)` : la colonne Pronostic passe en dernière position de l'affichage. App stable 5/5 en 200. Vérifié en ligne (CAN round 23:19). 507 tests |
| Tableau round enrichi | **FAIT** (16/10) — dernier `b28a1cf`, HF `01a6367`. Colonnes 1/X/2 + Over/Under 2,5 retirées de l'affichage. Emojis cibles 🔥 (classement) / 💥 (attaque, ex-🧊) / ⚡ (forme). Annotation 1-1 CAN : 🔒 (ex-🧱, deux 0-0) · 💀 (deux perdu sans marquer) · 🔻/🔻🔻 · 🔺/🔺🔺, sur les 3 scores (principal + 2 « Sinon »). Pour les matchs portant un emoji : colonne « Classement » (ex « 1er vs 4e ») + 5 derniers matchs en pastilles sous le tableau (`classement_saison`, `formes_recentes`, `tableau_round` renvoie `marques`). Que jouer : filtres Cote/Conseil retirés, détail = score exact/1X2/G-NG. Débusqueur effacé (2 onglets). Vérifié en ligne : Egypt 🔥💥⚡, Cameroon-Mozambique 1-1 💀. 507 tests |
| Scraper 30s + 1-1 en « Sinon » | **FAIT** (15/10) — dernier `1ebd0b6`, HF `c933448`. Scraper à **30 s** (était 180 s) dans `start_cloud.sh` : rounds toutes les 120 s, donc 30 s nécessaire pour que le résultat précédent soit en base 30 s avant le coup d'envoi suivant (60 s était trop juste, ~20 s de marge). L'annotation 1-1 porte aussi sur les 2 scores de « Sinon » (`_marque_11`). Base préservée au redéploiement (données à jour 15/10). **À SURVEILLER : ~6x plus de requêtes API qu'à 180 s → si le Space souffre ou l'API limite, remonter à 60 s** |
| Annotation 1-1 CAN | **FAIT** (15/10) — `5deb28b`, HF `532bd9d`. Sur un pronostic 1-1 en CAN, selon le dernier match de chaque équipe : 🧱 (deux 0-0), 🔻/🔻🔻 (dom 0-0, ext battu / 0-1), 🔺/🔺🔺 (ext 0-0, dom vainqueur / 1-0). `derniers_resultats`, `emoji_nul_11`, `LIGUES_ANNOTE_11={CAN}`. 494 tests. En ligne : légende visible, mais le round testé n'avait pas de 1-1 remplissant une condition (à revérifier sur un round qui en a un) |
| Poussé | **OUI** (04/10) — `origin` à jour (dernier `0ab0195` keep-alive + fraîcheur) |
| Déployé | **OUI** (04/10) — commit HF `b936761`. En ligne, la collecte est REPARTIE : bandeau « 🟢 Données à jour — dernier résultat 04/10/2026 19:41 » (la base n'est plus figée au 19/07). Le keep-alive (ping 12 h) maintiendra ça. À lancer une fois à la main : GitHub Actions → « Garder le Space HF éveillé » |
| Gel face-à-face (19/07) | Cause RÉELLE prouvée le 04/10 : pas un bug de code. Space HF dort après 48 h (gcTimeout) → scraper arrêté ; + pas de disque persistant → rebuild repart du seed du 5 juillet. API `/results` OK (6/9 ligues), parser OK. 3 ligues (ANG/ITA/ESP) en nouvelle saison = `/results` vide (normal). Voir mémoire `collecte-resultats-hf`. Correctif `0ab0195` : keep-alive GitHub Action (ping 12 h) + bandeau fraîcheur. Persistance aux rebuilds NON résolue (Olivio : « pas de préférence ») |
| Bug `p_h_bl` | **corrigé** dans `d113c64` sur demande d'Olivio : le 1X2 V2 du round lit désormais `p_h_blend` |
| « Que jouer ? » | **ne pas y toucher** (Olivio, 04/10) — le tableau est un onglet à part |
| Face-à-face | **ne plus y toucher** (Olivio, 04/10). Patch mis de côté pour ses ✅ et 🟢⚪🔴 : `h2h-trois-couleurs*.patch` dans le scratchpad de session |
| Non committé | les 3 `scraper/*.py` en CRLF, ce fichier, `.claude/launch.json` (serveur local) |
| Suite de tests | **405 passés** (04/10) |

Mis à jour le 04/10 :
- le garde-fou laissé à la session d'avant ne se chargeait même pas
  (chaîne cassée en deux : SyntaxError) — le code 66 venait de là ou de la
  machine, peu importe : corrigé ;
- son « test de détection » ne testait rien (il vérifiait qu'un ensemble
  privé d'un élément ne contenait plus cet élément) — il rejoue maintenant
  l'oubli réel, et un mutant qui aveugle l'analyseur le fait passer au rouge ;
- réparé, il a trouvé `scripts/ui_theme.py` hors liste : présent sur le
  Space et identique au dépôt, donc sans casse aujourd'hui, ajouté quand même ;
- Space contrôlé : Dockerfile, requirements, start_cloud.sh et les 6 modules
  `scraper/` que l'app atteint sont identiques à HEAD (aux fins de ligne près).

⚠️ Les trois fichiers `scraper/*.py` qui apparaissent modifiés sont du **CRLF
seul** — vérifié avec `git diff --ignore-cr-at-eol`. **Ne jamais les committer.**

## 3. CE QUI RESTE À FAIRE, DANS L'ORDRE

1. ~~Relancer la suite de tests~~ — fait, 403 passés.
2. ~~Committer~~ — fait, `70f6a9d`.
3. ~~Pousser et déployer~~ — fait le 04/10 sur GO d'Olivio.

### Pourquoi le point 2 est bloquant

`scripts/forme_pastilles.py` est un module NEUF que `dashboard_trio.py`
importe. Il n'était **pas** dans `DEFAULT_FILES` de `deploy/deploy_hf.py` :
l'app en ligne serait morte à l'import — écran blanc, sans rien pour
comprendre. Le dépôt était vert, les 400 tests passaient.

Le garde-fou ajouté lit les imports RÉELS des scripts déployés et vérifie que
chacun de leurs voisins locaux figure dans la liste. Il porte son propre test
de détection, pour ne pas donner un vert trompeur.

## 4. LA DERNIÈRE FONCTIONNALITÉ LIVRÉE (commit 81ca475)

Sous chaque confrontation de l'onglet « Historique & face-à-face », les **5
derniers résultats de chaque équipe AVANT cette rencontre** : rond vert à
coche (victoire), rond rouge à croix (défaite), rond gris à tiret (nul).

Décisions prises, et pourquoi :

- **En HTML, pas en émoji.** Aucun émoji ne porte à la fois la couleur et le
  glyphe de la maquette : la coche verte existe en carré, la croix rouge n'a
  pas de rond, et il n'y a pas de rond gris à tiret. `unsafe_allow_html` était
  déjà utilisé par le thème de l'app.
- **Seuls V, N et D traversent** — c'est ce qui rend l'insertion sûre : rien
  venu de la base n'atteint le HTML. La ligne des noms d'équipes reste en
  markdown pour cette raison.
- **Le match est exclu de sa propre forme**, sinon on lirait le résultat qu'on
  prétend annoncer. Le face-à-face PRÉCÉDENT, lui, compte : il a été joué.
- **Deux requêtes, pas soixante** : l'historique de chaque équipe est chargé
  une fois puis découpé en mémoire.
- Les pastilles vivent dans `scripts/forme_pastilles.py`, **sans dépendance
  Streamlit** : les garder dans le tableau de bord faisait tomber pytest sans
  trace sur cette machine.

### Un piège mesuré au passage

Les rounds de ces ligues tombent **toutes les 2 minutes**, pas toutes les 35.
La forme d'avant une confrontation n'est donc PAS la confrontation précédente,
contrairement à ce que l'écart d'heures affichées laissait croire. J'ai failli
conclure à un bug qui n'existait pas — contrôle fait avant de parler.

## 5. L'APP EN LIGNE AUJOURD'HUI

🧭 **Que jouer ?** — les 11 marchés cotés par rencontre, chacun avec sa
fiabilité mesurée, plus les deux mi-temps (1X2 + score exact).
Filtres : cote min. du conseil (1,20) et conseil retenu (X2).

🔎 **Débusqueur 1X2 à cote 2** — les 5 prochains rounds de chaque ligue, les
rencontres dont les TROIS cotes commencent par 2 (2,00 à 2,99).

🔎 **Historique & face-à-face** — deux équipes au choix, 9 ligues, avec les
pastilles de forme d'avant chaque confrontation.

**La prédiction du round** (section restaurée, hors onglet) — fit V5+V2
paresseux, 60 à 90 s au premier clic. Score exact sur TOUTES les rencontres.

📈 **Suivi réel (forward)**

## 6. RÈGLES DE TRAVAIL À NE PAS PERDRE

- **Jamais de push ni de déploiement sans GO explicite d'Olivio.**
- Bet261 pousse **uniquement** sur `DiabloBsr/bet`.
- La base locale est **figée au 5 juillet** : aucune rencontre à venir, donc
  les onglets ne rendent rien en local. Toute vérification passe par des
  matchs passés ou une base simulée (monkeypatch de `_upcoming_df`).
- Les heredocs Bash sont **tronqués au-delà d'environ 160 lignes** : écrire
  les gros fichiers en plusieurs morceaux.
- `.venv/Scripts/python.exe` est un Python **Windows** : lui donner des
  chemins **relatifs**, jamais `/d/projets/...`.
- Tout nouveau module importé par l'app doit entrer dans `DEFAULT_FILES`.
- Finir chaque session prédicteur par 2 à 4 pistes d'amélioration.

## 7. CE QUE LE MOTEUR SAIT, ET CE QU'IL NE PROUVE PAS

Taux « touché » = part des matchs où la sélection la plus probable est sortie
juste, mesurée hors échantillon d'entraînement.

| Marché | Touché | | Marché | Touché |
|---|---|---|---|---|
| Double Chance | 78,1 % | | Pair/Impair | 51,1 % |
| +/- 3,5 | 68,2 % | | 1X2 | 50,4 % |
| Multi-Buts | 66,5 % | | Minute du 1er but | 31,0 % |
| G/NG | 58,8 % | | HT/FT | 31,0 % |
| FTTS | 57,2 % | | Total de buts | 24,4 % |
| | | | Score exact | 11,8 % |

Mi-temps (non cotées par le book, calibrées à part sur 103 714 matchs) :
1re MT 1X2 44,4 % · 1re MT score exact 29,4 % · 2e MT 1X2 42,4 % ·
2e MT score exact 19,6 %.

⚠️ **Un taux élevé n'est pas un gain.** Le book price tout : ROI ≈ **−7 %**
sur le marché le mieux touché comme sur les autres.

⚠️ **La règle « proba × cote » est un signal INVERSÉ**, démontré deux fois sur
l'historique complet. Elle sélectionne les matchs où le modèle s'écarte le
plus du book, c'est-à-dire ses propres erreurs. Le bloc restauré de la section
du round l'utilise encore pour SÉLECTIONNER, mais le mot « value » a été
retiré et la mention dit ce qui a été mesuré. Ne pas le remettre.

## 8. TROIS PISTES OUVERTES

1. **Mesurer ce qui tourne sans preuve.** Trois choses reposent sur des règles
   plausibles mais jamais vérifiées : les signaux de piège (favori fragile,
   nul menaçant, aucun favori net), le bloc « proba × cote », et l'hypothèse
   implicite qu'un match aux trois cotes voisines serait plus prévisible.
   Une seule campagne — backtest sans fuite sur 207 000 matchs — répondrait
   aux trois. Si un signal ne tient pas, le dire plutôt que de le laisser
   rassurer à tort.
2. **Les filtres de « Que jouer ? »** masquent les 11 marchés sur 10
   rencontres sur 12. Les faire porter sur le seul conseil de tête.
3. **`tesseract` dans le Dockerfile**, inutile depuis la coupe des onglets
   OCR : du poids mort à chaque build.

## 9. FONCTIONS MOTEUR CONSERVÉES MAIS NON CÂBLÉES

Gardées volontairement : elles ne coûtent rien tant que rien ne les appelle,
et remettre un écran ne demandera alors que son affichage.

- `filtrer_equilibres` — les trois cotes au moins X
- `debusquer_cotes` — cible + tolérance sur les trois cotes
- `round_1x2` — appelée par le débusqueur, plus par un onglet propre
- `over25_scan`, `totals_scan`, `ou25_picks`, `resultats_a_cette_cote`,
  `associer_equipes`, module OCR `lire_cotes` — onglets supprimés le 27/09
- `tableau_heure`, `tableau_affichage`, `tableau_reco`, `_1x2_somme_100` —
  onglet « 📋 Tableau des prédictions » (1X2 à 100 %, O/U 2,5 et 3,5,
  pronostic en vert), créé puis retiré le 04/10 à la demande d'Olivio
