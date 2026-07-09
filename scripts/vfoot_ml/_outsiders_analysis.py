"""ALGORITHME OUTSIDERS — comment se comportent les équipes outsiders ?

Test clé : BIAIS FAVORI-OUTSIDER. Pour chaque sélection 1X2, on compare le taux
de réussite RÉEL à la proba implicite (dévigée), par tranche de proba. On regarde :
  - la courbe de calibration dans les tranches basses (outsiders) : sur/sous-cotés ?
  - le ROI par tranche aux VRAIES cotes (quelle tranche perd le moins / gagne ?)
  - quand un outsider gagne : domicile/extérieur, mène à la MT, etc. (descriptif)
Split chrono 70/30 pour vérifier que tout biais tient hors-échantillon.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT o.odds_home oh,o.odds_draw od,o.odds_away oa, r.score_a sa,r.score_b sb,
           r.ht_score_a ha, r.ht_score_b hb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1 ORDER BY e.expected_start"""), eng)
n = len(df); cut = int(n*0.7)
print(f"{n} matchs | train {cut} / test {n-cut}", flush=True)

# 3 sélections par match : (proba implicite dévigée, cote réelle, gagné?, venue, mène_MT)
inv = 1/df.oh + 1/df.od + 1/df.oa
recs = []
for i, r in enumerate(df.itertuples()):
    iv = 1/r.oh + 1/r.od + 1/r.oa
    outs = [((1/r.oh)/iv, r.oh, int(r.sa > r.sb), "dom", int(r.ha > r.hb)),
            ((1/r.od)/iv, r.od, int(r.sa == r.sb), "nul", int(r.ha == r.hb)),
            ((1/r.oa)/iv, r.oa, int(r.sb > r.sa), "ext", int(r.hb > r.ha))]
    for p, o, w, v, htlead in outs:
        recs.append((i, p, o, w, v, htlead))
S = pd.DataFrame(recs, columns=["i", "imp", "odds", "win", "venue", "htlead"])
S["train"] = S.i < cut

# ===== BIAIS FAVORI-OUTSIDER : réel vs implicite par tranche =====
bins = [0, .05, .10, .15, .20, .30, .40, .50, .65, .80, 1.01]
lbl = ["<5%", "5-10", "10-15", "15-20", "20-30", "30-40", "40-50", "50-65", "65-80", "80%+"]
S["b"] = pd.cut(S.imp, bins=bins, labels=lbl, right=False)
print("\n=== CALIBRATION PAR TRANCHE (biais favori-outsider ?) ===")
print(f"  {'tranche':<8}{'n':>7}{'implicite':>11}{'réel':>9}{'écart':>9}{'ROI réel':>10}{'ROI OOS':>10}")
for L in lbl:
    tr = S[(S.b == L) & S.train]; te = S[(S.b == L) & ~S.train]
    if len(tr) < 100: continue
    imp = tr.imp.mean(); real = tr.win.mean(); roi = (tr.win*tr.odds - 1).mean()
    roi_te = (te.win*te.odds - 1).mean() if len(te) > 50 else float("nan")
    print(f"  {L:<8}{len(tr):>7}{100*imp:>10.1f}%{100*real:>8.1f}%{100*(real-imp):>+8.1f}{100*roi:>+9.1f}%{100*roi_te:>+9.1f}%")
print("  écart>0 = l'issue gagne PLUS que la cote ne dit (sous-cotée) ; <0 = sur-cotée.")

# ===== FOCUS OUTSIDERS (imp<30%) : quand gagnent-ils ? =====
o = S[S.imp < 0.30]
print(f"\n=== OUTSIDERS (proba <30%) : {len(o)} sélections, réussite {100*o.win.mean():.1f}% ===")
print("  par POSITION :")
for v in ("dom", "nul", "ext"):
    s = o[o.venue == v]
    print(f"    {v:<4}: {len(s):>5} | réussite {100*s.win.mean():.1f}% (implicite {100*s.imp.mean():.1f}%) "
          f"| ROI {100*(s.win*s.odds-1).mean():+.1f}%")
print("  quand un outsider MÈNE à la mi-temps, gagne-t-il le match ?")
lead = o[o.htlead == 1]
print(f"    outsider menant MT : {len(lead)} | gagne le match {100*lead.win.mean():.0f}% "
      f"(vs {100*o.win.mean():.0f}% sans condition)")

print("\n" + "="*60)
print("  Si écart≈0 partout et ROI négatif partout -> outsiders CALIBRÉS :")
print("  pas de biais, parier gros = perdre la marge (souvent +, à cause de la")
print("  variance des grosses cotes). Aucun 'algorithme outsider' ne gagne.")
print("  Si une tranche a écart>0 stable ET ROI_OOS>0 -> vrai biais à exploiter.")
print("="*60)
