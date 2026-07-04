"""VFoot-ML — PHASE 4 : Backtesting de stratégies de paris.

Simule des paris 1X2 sur le test OOS, en utilisant les probabilités d'un modèle
(par défaut LogReg, notre meilleur) confrontées aux VRAIES cotes offertes.

Stratégies de mise : flat · Kelly fractionné · proportionnelle au bankroll.
Filtres : seuil de value (EV>X%), bande de cote, nb max de paris/jour.
Métriques : ROI, P/L cumulé, hit rate, drawdown max, Sharpe, courbe de bankroll.
Monte-Carlo : 10 000 scénarios -> probabilité de ruine + IC du profit.

Vérité attendue : aucune stratégie n'est +EV (le modèle ne bat pas la cote).
Le backtester est néanmoins l'organe de décision pour une future fenêtre de dérive.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_models import load, split, ALL, MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("vfoot.backtest")
rng = np.random.RandomState(42)
OUTCOME_COLS = ["odd_1", "odd_x", "odd_2"]


class Backtester:
    """Simulateur de paris 1X2. Toutes les méthodes sont vectorisées/sûres."""

    def __init__(self, proba: np.ndarray, odds: np.ndarray, y_true: np.ndarray):
        """proba (n,3) probas du modèle ; odds (n,3) cotes offertes ; y_true (n,) 0/1/2."""
        self.p = proba
        self.o = odds
        self.y = y_true
        self.ev = proba * odds - 1.0                 # espérance par issue

    def select(self, value_thr=0.0, odd_min=1.0, odd_max=100.0):
        """Sélectionne, par match, l'issue de meilleure EV si elle passe les filtres."""
        pick = self.ev.argmax(axis=1)
        ev_best = self.ev[np.arange(len(self.p)), pick]
        odd_pick = self.o[np.arange(len(self.p)), pick]
        mask = (ev_best > value_thr) & (odd_pick >= odd_min) & (odd_pick <= odd_max)
        return pick, mask

    def run(self, strategy="flat", value_thr=0.0, odd_min=1.0, odd_max=100.0,
            bankroll0=100.0, kelly_frac=0.25, prop_frac=0.02):
        """Simule la stratégie. Retourne métriques + courbe de bankroll."""
        pick, mask = self.select(value_thr, odd_min, odd_max)
        idx = np.where(mask)[0]
        bank = bankroll0
        curve, returns, wins = [bank], [], 0
        for i in idx:
            k = pick[i]; odd = self.o[i, k]; p = self.p[i, k]
            if strategy == "flat":
                stake = 1.0
            elif strategy == "kelly":
                edge = p * odd - 1.0
                stake = max(0.0, kelly_frac * edge / (odd - 1.0)) * bank
            else:                                    # proportional
                stake = prop_frac * bank
            stake = min(stake, bank)                 # pas de mise à crédit
            won = (self.y[i] == k)
            pnl = stake * (odd - 1.0) if won else -stake
            bank += pnl
            wins += int(won)
            returns.append(pnl / stake if stake > 0 else 0.0)
            curve.append(bank)
            if bank <= 0:
                break
        returns = np.array(returns)
        staked = len(idx) if strategy == "flat" else None
        roi = (bank - bankroll0) / (len(idx)) if strategy == "flat" and len(idx) else (
            (bank - bankroll0) / bankroll0 if len(idx) else 0.0)
        curve = np.array(curve)
        dd = float((np.maximum.accumulate(curve) - curve).max()) if len(curve) else 0.0
        sharpe = float(returns.mean() / returns.std() * np.sqrt(len(returns))) \
            if len(returns) > 1 and returns.std() > 0 else 0.0
        return {"strategy": strategy, "value_thr": value_thr, "n_bets": int(len(idx)),
                "hit_rate": round(100 * wins / len(idx), 2) if len(idx) else None,
                "roi_pct": round(100 * float(returns.mean()), 2) if len(returns) else None,
                "final_bankroll": round(float(bank), 2),
                "max_drawdown": round(dd, 2), "sharpe": round(sharpe, 3),
                "_curve": curve, "_returns": returns}

    def monte_carlo(self, returns, bankroll0=100.0, n_sims=10000, stake=1.0):
        """Bootstrap des paris (flat) -> distribution du bankroll final + P(ruine)."""
        if len(returns) < 10:
            return {}
        pnl_unit = returns * stake                   # gain/perte par pari (mise 1)
        n = len(pnl_unit)
        finals = np.empty(n_sims); ruined = 0
        for s in range(n_sims):
            draw = pnl_unit[rng.randint(0, n, n)]
            path = bankroll0 + np.cumsum(draw)
            finals[s] = path[-1]
            if path.min() <= 0:
                ruined += 1
        return {"n_sims": n_sims, "p_ruine_pct": round(100 * ruined / n_sims, 2),
                "profit_median": round(float(np.median(finals) - bankroll0), 2),
                "profit_IC95": [round(float(np.percentile(finals, 2.5) - bankroll0), 2),
                                round(float(np.percentile(finals, 97.5) - bankroll0), 2)]}


def main():
    df = load(); tr, te = split(df)
    # modèle : LogReg (notre meilleur, ~= la cote) entraîné sur le train
    model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=600))])
    model.fit(tr[ALL], tr.y)
    proba = model.predict_proba(te[ALL])
    odds = te[OUTCOME_COLS].to_numpy()
    y = te.y.to_numpy()
    bt = Backtester(proba, odds, y)

    configs = [
        ("flat", 0.0, 1.0, 100.0), ("flat", 0.02, 1.0, 100.0), ("flat", 0.05, 1.0, 100.0),
        ("kelly", 0.02, 1.0, 100.0), ("proportional", 0.05, 1.0, 100.0),
        ("flat", 0.05, 1.0, 3.0),         # value + favoris (cote<=3)
    ]
    rows = []
    best = None
    for strat, vt, omn, omx in configs:
        r = bt.run(strategy=strat, value_thr=vt, odd_min=omn, odd_max=omx)
        rows.append(r)
        if r["n_bets"] and (best is None or r["n_bets"] > best["n_bets"]):
            best = r

    mc = bt.monte_carlo(best["_returns"]) if best else {}

    # courbe de bankroll (config flat sans filtre)
    try:
        c = rows[0]["_curve"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(c, color="#e76f51"); ax.axhline(100, ls="--", color="k", alpha=.4)
        ax.set_title("Bankroll — flat, sans filtre (test OOS)")
        ax.set_xlabel("paris"); ax.set_ylabel("bankroll")
        fig.tight_layout(); fig.savefig("data/vfoot_ml/plots/06_bankroll.png", dpi=110); plt.close(fig)
    except Exception as e:
        logger.warning("plot bankroll: %s", e)

    out = {"n_test": len(te),
           "strategies": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
           "monte_carlo_best": mc}
    Path("data/vfoot_ml/phase4_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    print("  PHASE 4 — BACKTEST (test OOS, modèle LogReg vs vraies cotes 1X2)")
    print("=" * 78)
    print(f"  {'stratégie':<14}{'value':>7}{'n_paris':>9}{'hit%':>7}{'ROI%':>8}{'bankroll':>10}{'DDmax':>8}{'Sharpe':>8}")
    print("  " + "-" * 74)
    for r in rows:
        band = " (cote<=3)" if r["value_thr"] == 0.05 and r is rows[-1] else ""
        print(f"  {r['strategy']:<14}{r['value_thr']:>7}{r['n_bets']:>9}{str(r['hit_rate']):>7}"
              f"{str(r['roi_pct']):>8}{r['final_bankroll']:>10}{r['max_drawdown']:>8}{r['sharpe']:>8}{band}")
    if mc:
        print(f"\n  MONTE-CARLO (10 000 scénarios, bankroll 100u, mise 1u) :")
        print(f"    Probabilité de RUINE : {mc['p_ruine_pct']}%")
        print(f"    Profit médian : {mc['profit_median']}u | IC95% : {mc['profit_IC95']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
