# THÉORIES EN LIGNE — testées sur InstantLeague-8035 (Bet261/Sporty-Tech)

Toutes les « méthodes » qui circulent (FB, YouTube, Medium, Scribd, forums), passées
au crible : split OOS chrono, BH-FDR, ROI aux vraies cotes, IC95 bootstrap, contrôle
par permutation. Data : ~250 000 matchs, 9 ligues.

| Théorie (source) | Claim | Verdict | Chiffre |
|---|---|---|---|
| **Cycle seedé** (Medium/wintips) | 5 unders → under plus probable | ⚠️ empreinte réelle, **inexploitable** | +2.6pp OOS mais ROI −2.8% (marge) — **surveillé** |
| **Équilibrage** (wintips) | après overs → under dû | ❌ FAUX | EDGE −0.3 à −0.8pp |
| **3-4-3** (bouche à oreille) | nuls aux extrémités, buts/upsets au milieu | ❌ superstition | tout plat, χ²=3.8 (ns) |
| **Pair/impair** | la parité des buts alterne | ❌ i.i.d. | répétition 50.2% |
| **BTTS « safe »** (22bet) | marché le plus prévisible/rentable | ❌ calibré | réel 51.2% = implicite 51.2% |
| **Under 2.5 défensif = value** (footballpark) | Under 2.5 à cote 1.9-2.2 = value | ❌ FAUX | ROI −8.7% (comme toutes les bandes) |
| **Double chance rentable** (multiple) | DC = profit régulier | ❌ | ~74% mais ROI −marge (calibré) |
| **Suivre le favori** | favori = gain | ❌ | 55.8% = implicite, ROI −marge |
| **Momentum meter** (Medium) | ratio 10 derniers prédit | ❌ gambler's fallacy | sans mémoire prouvé |
| **Fréquence buts par ligue** | certaines ligues + prolifiques | ✅ vrai MAIS pricé | dans les cotes, 0 edge |
| **Over sous-pricés / mode drama** (plan interne) | simulateur surproduit → Over = value, surtout matchs serrés | ❌ FAUX | Over 3.5 ROI −5.7% = **la marge exacte** ; serrés −6.3%, déséq. −4.8% ; tout calibré |
| **Canal moins cher** (math) | exprimer une vue sur le marché à plus faible marge | ✅ vrai levier | Under 2.5 (9%) −9.3% → Under 3.5 (5.7%) −5.2% : perte /2, mais IC95 toujours < 0 |
| **Outsider à la mi-temps** (intuition user) | parier l'outsider à grosse cote sur le résultat MT = + rentable (peu importe la suite) | ❌ FAUX, pire | ROI −8.7% ; marge MT ~8-9% > marge FT 5.7% ; outsider mène 16.6% mais taux réel < implicite dans **chaque** bande de cote (surcôté) |
| **Paris mi-temps en général** (verdict complet) | un marché HT serait + rentable/prévisible | ❌ FAUX, tous pires que FT | Marges : BTTS-1H 6%, Mi-tps 1X2/DC 8%, **HT/FT 12% (pire du board)**. ROI = −marge partout (calibré) : favori-MT −7.0%, DC-MT −7.2%, nul-MT −7.6%, HT/FT outsider −11.7%. Meilleur HT (« 0 but en 1re période » −5.4%) même pas > meilleur FT. **Fuir le HT/FT.** |
| **Combiner des petits % → gros %** (intuition user) | assembler N marchés « faibles » rend rentable | ❌ FAUX, l'inverse | Aucun marché n'a d'edge positif (meilleur = suivre le favori −5.3%). Les edges se **multiplient** pas s'additionnent : ROI combiné = ∏(1+edgeᵢ)−1. Preuve empirique : favori 1 leg −4.8% → 2 legs −10.3% → 5 legs −19.4% → **10 legs −44.3%** (P(gagner) 54%→0.2%). Combiner des edges NÉGATIFS empile la marge — c'est pourquoi le book adore les combinés. |
| **Scores exacts rares à grosse cote** (intuition user) | viser 0-0 (<0.5), 2-2, 3-3, 3-2, 2-3, 3-1, 1-3, 4-1, 1-4… car grosses cotes | ❌ FAUX, le pire marché | **28 scores testés OOS, ZÉRO à IC95>0.** Tous overpricés (taux réel < implicite partout). ROI : 0-0 −10.4%, 1-1 −11.3%, 2-2 −12.5%, 3-3 −15.3%, jusqu'à 0-6 −19.3%. Le « boost 2-2 +31% » du simulateur est déjà dans le prix (et sur-pricé). Marché Score exact = 24% de marge. Table mesurée : `data/vfoot_ml/score_exact_roi.json`. |
| **Panier de SIMPLES (favoris + outsiders)** (intuition user) | 3-5 paris simples d'un coup mêlant favoris ~2.0 et outsiders gros cote | ❌ pas rentable, mais ≠ combiné | EV se **moyenne** (pas multiplie) → mieux que le combiné, mais moyenne de négatifs = négatif. Favori simple −5.4%, outsider 2.5-12 −6.4% (variance 2×). Paniers : 3 favoris −5.2% (le moins mauvais, gagnant 47%) ; ajouter outsiders → −6.3% + P(gagnant) 47%→40%. Diversification réduit les swings, PAS la marge. Rejoint l'historique user −18.6% (chasse aux outsiders). |
| **Stake 2-3% + discipline** (tous) | money management | ✅ **le seul conseil valable** | ne gagne pas, limite la casse |

## Conclusion
Sur **toutes** les théories concrètes trouvées en ligne, **une seule** (le cycle seedé)
a un noyau statistique réel — et elle reste **inexploitable** car la marge du bookmaker
(9-10% sur Multi-Buts) est plus grosse que l'edge (+2.6pp). Toutes les autres sont des
mirages de perception (patterns vus dans du bruit) ou des conseils de gestion.

**Deux sentinelles** surveillent les seuls leads réels : mouvement de ligne + cycle seedé.
Alerte automatique si l'un franchit la marge.

*Registre vivant : toute nouvelle théorie croisée est testée et ajoutée ici.*
