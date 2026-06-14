"""Audit V5 — partie B : rho_ht, form_alpha (rolling), half_life, fix IPF '6+'.

Walk-forward 70/30 temporel, metriques OOS only. NE MODIFIE AUCUN fichier scraper/.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v2 import (
    _dc_tau, compute_home_away_strengths, estimate_rho_v2,
    poisson_score_grid, grid_to_1x2, market_score_grid,
)

O = ["1", "X", "2"]


def outcome_of(sa, sb):
    return "1" if sa > sb else ("X" if sa == sb else "2")


def tier_stats(name, rows):
    """rows: (p_max, pick, out, cote)"""
    acc = np.mean([pk == o for _, pk, o, _ in rows]) * 100
    print(f"[{name}] acc_oos={acc:.2f}% (n={len(rows)})")
    for t in (0.6, 0.7):
        sub = [(pk == o, c) for p, pk, o, c in rows if p >= t]
        if sub:
            wr = np.mean([w for w, _ in sub]) * 100
            roi = np.mean([w * (c - 1) - (1 - w) for w, c in sub]) * 100
            flag = "" if len(sub) >= 30 else " [INSTABLE]"
            print(f"    p>={t:.0%}: n={len(sub)} wr={wr:.1f}% roi={roi:+.1f}%{flag}")


def main():
    engine = create_engine(load_settings().db_url)
    df = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND o.odds_home IS NOT NULL
              AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start
    """, engine)
    df = df.reset_index(drop=True)
    n = len(df); cut = int(n * 0.70)
    train, oos = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"total={n} train={len(train)} oos={len(oos)}\n")

    # ---- base fit (prod-like: smoothing=5, no decay) ----
    home_s, away_s, mu_h, mu_a = compute_home_away_strengths(train, smoothing=5, half_life=None)
    rho_ft = estimate_rho_v2(train, home_s, away_s, mu_h, mu_a)

    # ---- 1. rho_ht estime sur scores HT ----
    ht_train = train[train.ht_score_a.notna()]
    ratio = float((ht_train.ht_score_a + ht_train.ht_score_b).sum()
                  / (ht_train.score_a + ht_train.score_b).sum())
    rows_ht = []
    for r in ht_train.itertuples():
        sh = home_s.get(r.team_a); sa = away_s.get(r.team_b)
        if sh is None or sa is None: continue
        lh = sh["attack"] * sa["defense"] * mu_h * ratio
        la = sa["attack"] * sh["defense"] * mu_a * ratio
        rows_ht.append((int(r.ht_score_a), int(r.ht_score_b), lh, la))

    def nll_ht(rho):
        ll = 0.0
        for h, a, lh, la in rows_ht:
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            tau = _dc_tau(h, a, lh, la, rho)
            if p > 0 and tau > 0:
                ll += np.log(p * tau)
        return -ll
    rho_ht = float(minimize_scalar(nll_ht, bounds=(-0.5, 0.5), method="bounded").x)
    print(f"rho_ft={rho_ft:.4f}  rho_ht(ML sur HT)={rho_ht:.4f}  ht_ratio={ratio:.3f}")

    ht_oos = oos[oos.ht_score_a.notna()]
    res = {"ind": [], "dc_ft": [], "dc_ht": []}
    for r in ht_oos.itertuples():
        sh = home_s.get(r.team_a); sa = away_s.get(r.team_b)
        if sh is None or sa is None: continue
        lh = sh["attack"] * sa["defense"] * mu_h * ratio
        la = sa["attack"] * sh["defense"] * mu_a * ratio
        out = outcome_of(r.ht_score_a, r.ht_score_b)
        for nm, rho in (("ind", 0.0), ("dc_ft", rho_ft), ("dc_ht", rho_ht)):
            p3 = np.zeros(3)
            for h in range(6):
                for a in range(6):
                    p = poisson.pmf(h, lh) * poisson.pmf(a, la) * _dc_tau(h, a, lh, la, rho)
                    p3[0 if h > a else (1 if h == a else 2)] += max(p, 0)
            res[nm].append((O[int(p3.argmax())], out))
    print("HT 1X2 OOS:")
    for nm, lab in (("ind", "Poisson independant (prod)"), ("dc_ft", "DC avec rho_ft"), ("dc_ht", "DC avec rho_ht")):
        acc = np.mean([a == b for a, b in res[nm]]) * 100
        print(f"  {lab:<30} acc={acc:.2f}% (n={len(res[nm])})")

    # ---- 2. form_alpha via rolling last-5 (proxy de rankings.history) ----
    print("\nFORM rolling last-5 (proxy rankings_snapshots), applique a la V3 boost formula:")
    from collections import deque, defaultdict
    last5 = defaultdict(lambda: deque(maxlen=5))
    # warm-up sur le train
    for r in train.itertuples():
        out = outcome_of(r.score_a, r.score_b)
        last5[r.team_a].append(3 if out == "1" else (1 if out == "X" else 0))
        last5[r.team_b].append(3 if out == "2" else (1 if out == "X" else 0))

    def form_of(team):
        d = last5[team]
        return (sum(d) / (3 * len(d))) if d else 0.5

    rows_by_alpha = {a: [] for a in (0.0, 0.1, 0.2, 0.3)}
    for r in oos.itertuples():
        sh = home_s.get(r.team_a); sa = away_s.get(r.team_b)
        out = outcome_of(r.score_a, r.score_b)
        if sh and sa:
            fa, fb = form_of(r.team_a), form_of(r.team_b)
            for alpha in rows_by_alpha:
                ba = (fa - 0.5) * alpha * 2
                bb = (fb - 0.5) * alpha * 2
                lh = sh["attack"] * (1 + ba) * sa["defense"] * max(0.5, 1 - bb) * mu_h
                la = sa["attack"] * (1 + bb) * sh["defense"] * max(0.5, 1 - ba) * mu_a
                g = poisson_score_grid(lh, la, rho_ft)
                p3 = np.array(grid_to_1x2(g))
                k = int(p3.argmax())
                cote = [r.odds_home, r.odds_draw, r.odds_away][k]
                rows_by_alpha[alpha].append((p3[k], O[k], out, cote))
        # update rolling APRES la prediction (anti-leak)
        last5[r.team_a].append(3 if out == "1" else (1 if out == "X" else 0))
        last5[r.team_b].append(3 if out == "2" else (1 if out == "X" else 0))
    for alpha, rows in rows_by_alpha.items():
        tier_stats(f"form_alpha={alpha}", rows)

    # ---- 3. half_life ----
    print("\nHALF-LIFE (time-decay) — poisson pur OOS:")
    for hl in (None, 500, 1500):
        hs, as_, mh, ma = compute_home_away_strengths(train, smoothing=5, half_life=hl)
        rho = estimate_rho_v2(train, hs, as_, mh, ma)
        rows = []
        for r in oos.itertuples():
            sh = hs.get(r.team_a); sa = as_.get(r.team_b)
            if not (sh and sa): continue
            lh = sh["attack"] * sa["defense"] * mh
            la = sa["attack"] * sh["defense"] * ma
            g = poisson_score_grid(lh, la, rho)
            p3 = np.array(grid_to_1x2(g))
            k = int(p3.argmax())
            cote = [r.odds_home, r.odds_draw, r.odds_away][k]
            rows.append((p3[k], O[k], outcome_of(r.score_a, r.score_b), cote))
        tier_stats(f"half_life={hl}", rows)

    # ---- 4. IPF '6+' fix : '6+' contraint h+a>=6 au lieu de ==6 ----
    print("\nIPF fix '6+' (cellules h+a>=6) — impact O2.5/O3.5 brier + acc:")

    def ipf(em, plus_as_tail: bool, max_goals=8, max_iter=30):
        sc = em.get("Score exact") if isinstance(em, dict) else None
        base = market_score_grid(sc, max_goals)
        if base is None: return None
        grid = base.copy()
        tot_market = em.get("Total de buts") if isinstance(em, dict) else None
        tot_target, tail_target = {}, None
        if isinstance(tot_market, dict):
            for k, cote in tot_market.items():
                try:
                    ks = str(k).strip()
                    nn = int(ks.rstrip("+").strip())
                    if ks.endswith("+") and plus_as_tail:
                        tail_target = (nn, 1.0 / float(cote))
                    else:
                        tot_target[nn] = 1.0 / float(cote)
                except (ValueError, TypeError):
                    continue
            s = sum(tot_target.values()) + (tail_target[1] if tail_target else 0)
            if s > 0:
                tot_target = {k: v / s for k, v in tot_target.items()}
                if tail_target: tail_target = (tail_target[0], tail_target[1] / s)
            else:
                tot_target, tail_target = None, None
        gng = em.get("G/NG") if isinstance(em, dict) else None
        btts_target = None
        if isinstance(gng, dict) and "Oui" in gng and "Non" in gng:
            try:
                po, pn = 1.0 / float(gng["Oui"]), 1.0 / float(gng["Non"])
                btts_target = po / (po + pn)
            except Exception: pass
        if not tot_target and btts_target is None:
            return grid
        idx = np.add.outer(np.arange(max_goals), np.arange(max_goals))
        for _ in range(max_iter):
            old = grid.copy()
            if tot_target:
                for ng, t in tot_target.items():
                    mask = idx == ng
                    cur = grid[mask].sum()
                    if cur > 0 and t > 0: grid[mask] *= t / cur
            if tail_target:
                mask = idx >= tail_target[0]
                cur = grid[mask].sum()
                if cur > 0 and tail_target[1] > 0: grid[mask] *= tail_target[1] / cur
            if btts_target is not None:
                m_b = (np.arange(max_goals)[:, None] >= 1) & (np.arange(max_goals)[None, :] >= 1)
                cb, cn = grid[m_b].sum(), grid[~m_b].sum()
                if cb > 0: grid[m_b] *= btts_target / cb
                if cn > 0: grid[~m_b] *= (1 - btts_target) / cn
            grid = np.clip(grid, 0, None)
            s = grid.sum()
            if s > 0: grid /= s
            if np.abs(grid - old).max() < 1e-4: break
        return grid

    idx8 = np.add.outer(np.arange(8), np.arange(8))
    stats = {False: {"o25": [], "o35": []}, True: {"o25": [], "o35": []}}
    y25, y35 = [], []
    nb = 0
    for r in oos.itertuples():
        em = r.extra_markets
        if isinstance(em, str):
            try: em = json.loads(em)
            except Exception: em = None
        if not isinstance(em, dict): continue
        g0 = ipf(em, plus_as_tail=False)
        if g0 is None: continue
        g1 = ipf(em, plus_as_tail=True)
        tot = int(r.score_a) + int(r.score_b)
        y25.append(float(tot >= 3)); y35.append(float(tot >= 4)); nb += 1
        for tail, g in ((False, g0), (True, g1)):
            stats[tail]["o25"].append(float(g[idx8 >= 3].sum()))
            stats[tail]["o35"].append(float(g[idx8 >= 4].sum()))
    y25 = np.array(y25); y35 = np.array(y35)
    for tail, lab in ((False, "prod  ('6+' traite ==6)"), (True, "fix   ('6+' traite >=6)")):
        p25 = np.array(stats[tail]["o25"]); p35 = np.array(stats[tail]["o35"])
        print(f"  {lab}: O2.5 acc={(np.round(p25)==y25).mean()*100:5.1f}% brier={np.mean((p25-y25)**2):.4f} | "
              f"O3.5 acc={(np.round(p35)==y35).mean()*100:5.1f}% brier={np.mean((p35-y35)**2):.4f} (n={nb})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
