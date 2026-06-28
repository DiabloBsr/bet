"""MONITEUR DE DÉRIVE v2 — multi-flux (global + par ligue + par bande de cote),
alarme à seuil + CONFIRMATION par la fenêtre précédente, taux de faux positifs explicite.
Surveille la calibration glissante (réel - implicite). Pas de prédiction.
Usage: ./.venv/Scripts/python.exe scripts/_drift_monitor_v2.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine
from scraper.config import load_settings

W = 2000; Z_ALARM = 3.0; Z_CONFIRM = 2.0; MINW = 500

e = create_engine(load_settings().db_url)
d = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""", e)
d = d[(d.oh > 1) & (d.od > 1) & (d.oa > 1)].copy()
d["es"] = pd.to_datetime(d.expected_start, utc=True, errors="coerce")
d = d.dropna(subset=["es"]).sort_values(["es", "comp", "ev"]).reset_index(drop=True)
inv = 1/d.oh + 1/d.od + 1/d.oa
d["imp_home"] = (1/d.oh)/inv; d["imp_away"] = (1/d.oa)/inv; d["imp_draw"] = (1/d.od)/inv
d["p_fav"] = d[["imp_home", "imp_away"]].max(axis=1)
d["fav_home"] = d.imp_home > d.imp_away
d["fav_won"] = np.where(d.fav_home, d.sa > d.sb, d.sb > d.sa).astype(float)
d["r_fav"] = d.fav_won - d.p_fav
d["league"] = d.comp.str.replace("InstantLeague-", "", regex=False)
res = np.where(d.sa > d.sb, d.imp_home, np.where(d.sa == d.sb, d.imp_draw, d.imp_away))
d["ll"] = -np.log(np.clip(res, 1e-9, 1))

def stat(frame):
    if len(frame) < MINW: return None
    cur = frame.r_fav.iloc[-min(W, len(frame)):]
    zc = cur.mean()/(cur.std()/math.sqrt(len(cur))) if cur.std() > 0 else 0
    if len(frame) >= 2*W:
        prev = frame.r_fav.iloc[-2*W:-W]
        zp = prev.mean()/(prev.std()/math.sqrt(len(prev))) if prev.std() > 0 else 0
    else:
        zp = 0.0
    return dict(n=len(cur), pp=cur.mean()*100, zc=zc, zp=zp, ll=frame.ll.iloc[-min(W, len(frame)):].mean())

def stt(s):
    if s is None: return "n/a"
    if abs(s["zc"]) > Z_ALARM and np.sign(s["zc"]) == np.sign(s["zp"]) and abs(s["zp"]) > Z_CONFIRM: return "ALARME"
    if abs(s["zc"]) > Z_ALARM: return "WATCH"
    return "OK"

streams = {"GLOBAL": d}
for lg, g in d.groupby("league"): streams[f"ligue {lg}"] = g
for lab, lo, hi in [("p_fav 0.40-0.50", .40, .50), ("p_fav 0.50-0.65", .50, .65), ("p_fav 0.65-1.0", .65, 1.01)]:
    streams[lab] = d[(d.p_fav >= lo) & (d.p_fav < hi)]

p_fp = 2*(1-norm.cdf(Z_ALARM)) * (2*(1-norm.cdf(Z_CONFIRM)))
ns = sum(1 for s in streams.values() if len(s) >= MINW)
print("="*76)
print("MONITEUR DÉRIVE v2 — logique & faux positifs")
print("="*76)
print(f"  W={W} | ALARME si |z_cur|>{Z_ALARM} ET même signe + |z_prev|>{Z_CONFIRM}")
print(f"  flux={ns} (global+ligues+bandes) | P(FP)/flux/check ≈ {p_fp:.1e}")
print(f"  P(>=1 FP sur {ns} flux) ≈ {1-(1-p_fp)**ns:.1e}  -> ~1 fausse alarme tous les {int(1/(1-(1-p_fp)**ns)):,} checks")
print("\n" + "="*76)
print(f"DASHBOARD — dernière donnée {d.es.max():%Y-%m-%d %H:%M} UTC")
print("="*76)
print(f"  {'flux':<20}{'n':>6}{'résidu pp':>11}{'z_cur':>7}{'z_prev':>8}{'logloss':>9}  statut")
print("  " + "-"*70)
for nm, fr in streams.items():
    s = stat(fr)
    if s is None: print(f"  {nm:<20}{'<min':>6}"); continue
    print(f"  {nm:<20}{s['n']:>6}{s['pp']:>+11.2f}{s['zc']:>+7.2f}{s['zp']:>+8.2f}{s['ll']:>9.4f}  {stt(s)}")
al = [nm for nm, fr in streams.items() if stt(stat(fr)) == "ALARME"]
print("\n" + "="*76)
print(f"VERDICT : {len(al)} alarme(s)  ->", al if al else "aucune dérive, système stable sur tous les flux.")
