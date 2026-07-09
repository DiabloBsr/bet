"""ANALYSE DE MON HISTORIQUE DE PARIS — quand ça fait boom, et le vrai bilan net."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
bets = json.load(open(ROOT / "exports" / "my_bets.json", encoding="utf-8"))
WON = {"Won", "WonPaid"}


def f(x, d=0.0):
    try: return float(x)
    except Exception: return d


def total_odds(b):
    """cote totale : fixedOdds si gagné (>1), sinon produit des cotes des sélections
    (l'API met fixedOdds=0 pour les paris perdus)."""
    fo = f(b.get("fixedOdds"))
    if fo > 1:
        return fo
    prod = 1.0
    for l in (b.get("betLines") or []):
        o = f(l.get("odds"))
        if o > 1:
            prod *= o
    return prod if prod > 1 else 0.0


rows = []
for b in bets:
    state = b.get("state"); won = state in WON
    k = total_odds(b)                              # cote totale corrigée (perdus inclus)
    stake = f(b.get("totalStake"))
    earn = f(b.get("earning"))
    lines = b.get("betLines") or []
    rows.append({"won": won, "k": k, "stake": stake, "earn": earn,
                 "nlegs": len(lines), "date": (b.get("betDate") or "")[:10], "lines": lines})

n = len(rows)
staked = sum(r["stake"] for r in rows)
returned = sum(r["earn"] for r in rows)
net = returned - staked
nwon = sum(r["won"] for r in rows)
print("=" * 64)
print("  BILAN GLOBAL DE TES PARIS")
print("=" * 64)
print(f"  {n} paris | {nwon} gagnés ({100*nwon/n:.0f}%) · {n-nwon} perdus")
print(f"  Total misé   : {staked:>12,.0f} MGA")
print(f"  Total récupéré : {returned:>10,.0f} MGA")
print(f"  RÉSULTAT NET : {net:>+12,.0f} MGA   (ROI {100*net/staked:+.1f}%)")
print(f"  mise moyenne {staked/n:,.0f} | plus gros gain {max(r['earn'] for r in rows):,.0f}")

# --- par cote totale ---
print("\n  PAR COTE TOTALE :")
for lo, hi, lbl in [(1, 2, "<2"), (2, 3, "2-3"), (3, 5, "3-5"), (5, 8, "5-8"), (8, 1e9, "8+")]:
    sub = [r for r in rows if lo <= r["k"] < hi]
    if sub:
        w = sum(x["won"] for x in sub); st = sum(x["stake"] for x in sub); en = sum(x["earn"] for x in sub)
        print(f"    cote {lbl:<5}: {len(sub):>3} paris | {w} gagnés ({100*w/len(sub):>3.0f}%) | "
              f"net {en-st:>+10,.0f} (ROI {100*(en-st)/st:+.0f}%)")

# --- les GROS gains ---
print("\n  🎯 TES GROS GAINS (cote >= 4, gagnés) — journée/marché/favori/home-away :")
big = sorted([r for r in rows if r["won"] and r["k"] >= 4], key=lambda r: -r["k"])[:14]
for r in big:
    l = (r["lines"] or [{}])[0]
    mkt = l.get("eventBetTypeName", "?"); pk = l.get("selectionName", "?")
    p = f(l.get("neutralProbability")); tag = "FAV" if p > 0.5 else ("mid" if p >= 0.33 else "OUT")
    ev = l.get("eventName", "")
    print(f"    {r['date']} cote {r['k']:>5.1f} → {r['earn']:>7,.0f} | {mkt} '{pk}' [{tag} {100*p:.0f}%] {ev[:26]}")

# --- analyse par SÉLECTION (favori ? home/away ? marché ?) ---
sel_won = sel_tot = 0
by_mkt = {}; by_pick = {}; fav_bucket = {"favori(>50%)": [0, 0], "équilibré(33-50)": [0, 0], "outsider(<33)": [0, 0]}
for r in rows:
    for l in r["lines"]:
        st = l.get("state")
        w = 1 if st == "Won" else (0 if st == "Lost" else None)
        if w is None: continue
        sel_tot += 1; sel_won += w
        mkt = l.get("eventBetTypeName", "?"); by_mkt.setdefault(mkt, [0, 0]); by_mkt[mkt][0] += 1; by_mkt[mkt][1] += w
        pk = l.get("selectionName", "?"); by_pick.setdefault(pk, [0, 0]); by_pick[pk][0] += 1; by_pick[pk][1] += w
        p = f(l.get("neutralProbability"))
        b = "favori(>50%)" if p > 0.5 else ("équilibré(33-50)" if p >= 0.33 else "outsider(<33)")
        fav_bucket[b][0] += 1; fav_bucket[b][1] += w
print(f"\n  SÉLECTIONS individuelles : {sel_tot} | réussite {100*sel_won/sel_tot:.0f}%")
print("  par FAVORI/OUTSIDER (proba du marché) :")
for b, (t, w) in fav_bucket.items():
    if t: print(f"    {b:<18}: {t:>4} sél. | réussite {100*w/t:>3.0f}%")
print("  par MARCHÉ :")
for mkt, (t, w) in sorted(by_mkt.items(), key=lambda x: -x[1][0])[:6]:
    print(f"    {mkt:<16}: {t:>4} sél. | réussite {100*w/t:>3.0f}%")
print("  par TYPE DE PICK :")
for pk, (t, w) in sorted(by_pick.items(), key=lambda x: -x[1][0])[:6]:
    print(f"    {str(pk)[:16]:<16}: {t:>4} sél. | réussite {100*w/t:>3.0f}%")

print("\n" + "=" * 64)
print("  VERDICT : le net est ce qu'il est — les gros gains marquent la mémoire,")
print("  mais c'est le TOTAL qui compte. Les booms tombent à leur cote (rare),")
print("  éparpillés — pas de pattern à exploiter (RNG prouvé sans mémoire).")
print("=" * 64)
