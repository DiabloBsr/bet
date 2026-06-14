# Narrow Decision Table — chaînage total × dominance × BTTS -> 1 score

- n=9051 test=2716

Lecture : pour un match dont l'inversion donne (total attendu, qui mène, BTTS),
le score modal historique + son taux OOS + la cote offerte + l'EV.

```
total dominance     btts   n score  rate  cote    ev
   ~2     home+ BTTS-non  93   1-1  10.0   6.5 -38.0
 ~2-3    away++ BTTS-non 136   1-2  17.0   7.8  33.0
 ~2-3    away++ BTTS-oui  91   1-2  15.0   7.8  19.0
 ~2-3     away+ BTTS-oui 100   1-1  11.0   7.3 -21.0
 ~2-3      égal BTTS-oui 157   2-1  14.0   9.1  30.0
 ~2-3     home+ BTTS-oui 203   1-1  15.0   7.1   5.0
 ~2-3    home++ BTTS-non 162   1-1  12.0   8.5  -1.0
 ~2-3    home++ BTTS-oui 195   2-1  12.0   7.8  -5.0
   ~3    away++ BTTS-oui 115   1-2  14.0   7.9   9.0
   ~3     home+ BTTS-oui 121   2-1  12.0   8.3   4.0
   ~3    home++ BTTS-non 197   4-0   8.0   NaN   NaN
   ~3    home++ BTTS-oui 392   2-1  14.0   7.9   6.0
 ~3-4    home++ BTTS-non 128   3-0  10.0   7.6 -22.0
 ~3-4    home++ BTTS-oui 209   2-1  10.0   8.2 -17.0
```