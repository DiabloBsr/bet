"""DÉVIGAGE : proportionnel (actuel) vs SHIN vs POWER — meilleure proba de score ?

La marge du book n'est peut-être pas répartie uniformément (biais favori-outsider).
Shin/Power la retirent plus finement. Bat-on le Top-3 (31.5%) avec un meilleur dévig ?
Split chrono ; calibration 7x7 fit sur TRAIN. Score-exact market comme source.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}'"""), eng)
n = len(df); cut = int(n * 0.7)
print(f"{n} matchs | train {cut} / test {n-cut}", flush=True)


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


def parse_se(raw):
    try:
        xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return None
    se = gm(xm, "Score exact")
    if not isinstance(se, dict):
        return None
    d = {k.replace(":", "-").replace(" ", ""): float(v) for k, v in se.items()
         if isinstance(v, (int, float)) and 1 < v < 99.99}
    return d or None


def prop_devig(d):
    q = {s: 1/o for s, o in d.items()}; z = sum(q.values())
    return {s: v/z for s, v in q.items()}


def power_devig(d, k):
    q = {s: (1/o)**k for s, o in d.items()}; z = sum(q.values())
    return {s: v/z for s, v in q.items()}


def shin_devig(d):
    q = np.array([1/o for o in d.values()]); B = q.sum(); ss = list(d.keys())
    def sumpi(z):
        pi = (np.sqrt(z*z + 4*(1-z)*q*q/B) - z) / (2*(1-z)) if z < 1 else q/B
        return pi.sum(), pi
    lo, hi = 1e-6, 0.2
    for _ in range(40):
        mid = (lo+hi)/2; s, _ = sumpi(mid)
        if s > 1: lo = mid
        else: hi = mid
    _, pi = sumpi((lo+hi)/2)
    return dict(zip(ss, pi))


def top3(dist):
    return [s for s, _ in sorted(dist.items(), key=lambda kv: -kv[1])[:3]]


# calibration 7x7 sur TRAIN (fréquence empirique / proba moyenne proportionnelle)
sa6 = df.sa.clip(0, 6).astype(int).values; sb6 = df.sb.clip(0, 6).astype(int).values
emp = np.zeros((7, 7)); mod = np.zeros((7, 7)); nmod = 0
dists = [parse_se(r) for r in df.xm]
for i in range(cut):
    if dists[i]:
        emp[sa6[i], sb6[i]] += 1
        for s, p in prop_devig(dists[i]).items():
            try:
                h, a = map(int, s.split("-"))
                if h < 7 and a < 7: mod[h, a] += p
            except Exception: pass
        nmod += 1
emp /= emp.sum(); mod /= max(nmod, 1)
CAL = np.clip(emp / np.clip(mod, 1e-5, None), 0.4, 2.5)


def apply_cal(dist):
    out = {}
    for s, p in dist.items():
        try:
            h, a = map(int, s.split("-")); f = CAL[h][a] if (h < 7 and a < 7) else 1.0
        except Exception: f = 1.0
        out[s] = p * f
    z = sum(out.values()) or 1
    return {s: v/z for s, v in out.items()}


# fit k du power devig sur TRAIN (max Top-3)
best_k, best_h3 = 1.0, 0
for k in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2):
    h3 = 0; nn = 0
    for i in range(cut):
        if dists[i]:
            nn += 1
            if f"{sa6[i]}-{sb6[i]}" in top3(power_devig(dists[i], k)): h3 += 1
    if h3/max(nn, 1) > best_h3/max(1, 1): best_h3, best_k = h3, k
print(f"  power devig k* (fit train) = {best_k}", flush=True)

methods = {"proportionnel (actuel)": lambda d: prop_devig(d),
           "power (k*)": lambda d: power_devig(d, best_k),
           "shin": lambda d: shin_devig(d)}
res = {m: [0, 0, 0, 0] for m in methods}   # top1, top3, top1_cal, top3_cal
cnt = 0
for i in range(cut, n):
    if not dists[i]:
        continue
    cnt += 1; actual = f"{sa6[i]}-{sb6[i]}"
    for m, fn in methods.items():
        dist = fn(dists[i])
        t3 = top3(dist)
        res[m][0] += int(t3[0] == actual); res[m][1] += int(actual in t3)
        c3 = top3(apply_cal(dist))
        res[m][2] += int(c3[0] == actual); res[m][3] += int(actual in c3)

print(f"\n{cnt} matchs test\n{'méthode dévig':<24}{'Top-1':>8}{'Top-3':>8}{'Top-1+cal':>11}{'Top-3+cal':>11}")
for m, (h1, h3, h1c, h3c) in res.items():
    print(f"{m:<24}{100*h1/cnt:>7.2f}%{100*h3/cnt:>7.2f}%{100*h1c/cnt:>10.2f}%{100*h3c/cnt:>10.2f}%")
print("\n-> Si Shin/power > proportionnel de >0.5pp : on change le dévigage. Sinon : nul.")
