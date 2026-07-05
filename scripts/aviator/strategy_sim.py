"""Simulateur de STRATÉGIES de cash-out Aviator — les vrais EV / variance / risque de ruine.

Les manches sont i.i.d. (provably-fair) -> on ré-échantillonne la distribution EMPIRIQUE
des multiplicateurs collectés (Monte-Carlo) pour estimer, pour une stratégie donnée :
  ROI moyen, variance, drawdown max, probabilité de RUINE (bankroll -> 0).

Stratégies :
  fixed(T)       : mise constante, cash-out auto à T. Gagne stake*(T-1) si crash>=T, sinon perd stake.
  martingale(T)  : mise doublée après chaque perte (revient à la base après un gain), cash-out à T.

Vérité : à mise constante, EV/round = stake * (P(M>=T)*T - 1) = -stake*marge, QUEL QUE SOIT T.
Aucune stratégie ne change l'espérance ; elles ne changent que la forme du risque.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "aviator.db"
# générateur déterministe (pas de Math.random interdit ; seed fixe = reproductible)
_RNG = np.random.default_rng(12345)


def load_multipliers(db_path=DB) -> np.ndarray:
    db = sqlite3.connect(db_path)
    m = np.array([r[0] for r in db.execute("SELECT multiplier FROM aviator_rounds")], float)
    db.close()
    return m


def _draw(emp, k):
    """k multiplicateurs tirés de la distribution empirique (bootstrap)."""
    return emp[_RNG.integers(0, len(emp), size=k)]


def simulate(emp, strategy="fixed", target=2.0, stake=1000.0, bankroll=20000.0,
             rounds=200, sims=4000, mart_cap=None):
    """Monte-Carlo. Retourne stats agrégées + une trajectoire d'exemple."""
    if len(emp) < 10:
        return {"error": "pas assez de manches collectées"}
    finals, ruins, drawdowns, rois = [], 0, [], []
    example = None
    for s in range(sims):
        bk = bankroll
        peak = bk
        dd = 0.0
        cur_stake = stake
        traj = [bk]
        draws = _draw(emp, rounds)
        ruined = False
        for m in draws:
            bet = min(cur_stake, bk) if strategy == "martingale" else stake
            if bet <= 0 or bk <= 0:
                ruined = True; break
            if m >= target:                      # cash-out réussi
                bk += bet * (target - 1)
                cur_stake = stake                # reset martingale
            else:                                # crash avant la cible
                bk -= bet
                if strategy == "martingale":
                    cur_stake = min(cur_stake * 2, mart_cap or bankroll)
            peak = max(peak, bk)
            dd = max(dd, (peak - bk) / peak if peak > 0 else 0)
            traj.append(bk)
            if bk <= 0:
                ruined = True; break
        finals.append(max(bk, 0.0)); drawdowns.append(dd)
        rois.append((bk - bankroll) / bankroll)
        ruins += int(ruined or bk <= 0)
        if s == 0:
            example = traj
    finals = np.array(finals); rois = np.array(rois)
    # EV théorique par round (mise constante)
    p_win = float((emp >= target).mean())
    ev_round = stake * (p_win * target - 1)
    return {
        "strategy": strategy, "target": target, "stake": stake, "bankroll0": bankroll,
        "rounds": rounds, "sims": sims,
        "p_cashout": round(100 * p_win, 1),
        "ev_per_round": round(ev_round, 2),
        "ev_session_theo": round(ev_round * rounds, 1),
        "roi_mean_pct": round(100 * float(rois.mean()), 1),
        "roi_median_pct": round(100 * float(np.median(rois)), 1),
        "final_mean": round(float(finals.mean()), 0),
        "final_p5": round(float(np.percentile(finals, 5)), 0),
        "final_p95": round(float(np.percentile(finals, 95)), 0),
        "prob_ruin_pct": round(100 * ruins / sims, 1),
        "prob_profit_pct": round(100 * float((finals > bankroll).mean()), 1),
        "drawdown_mean_pct": round(100 * float(np.mean(drawdowns)), 1),
        "example_trajectory": [round(x, 0) for x in (example or [])[:rounds + 1]],
    }


def compare_targets(emp, targets=(1.3, 1.5, 2, 3, 5, 10), **kw):
    return {T: simulate(emp, "fixed", target=T, **kw) for T in targets}


if __name__ == "__main__":
    emp = load_multipliers()
    print(f"distribution empirique : {len(emp)} manches\n")
    print(f"{'cible':>6}{'P(cashout)':>12}{'ROI moy':>10}{'P(profit)':>11}"
          f"{'P(ruine)':>10}{'drawdown':>10}")
    for T, r in compare_targets(emp, rounds=200, sims=3000).items():
        if "error" in r:
            print(r["error"]); break
        print(f"{T:>6}{r['p_cashout']:>11}%{r['roi_mean_pct']:>9}%{r['prob_profit_pct']:>10}%"
              f"{r['prob_ruin_pct']:>9}%{r['drawdown_mean_pct']:>9}%")
    print("\nMartingale (cible 2.0) :")
    mg = simulate(emp, "martingale", target=2.0, rounds=200, sims=3000)
    if "error" not in mg:
        print(f"  ROI moy {mg['roi_mean_pct']:+}% | P(profit) {mg['prob_profit_pct']}% "
              f"| P(RUINE) {mg['prob_ruin_pct']}% | drawdown {mg['drawdown_mean_pct']}%")
    print("\n>>> EV identique quelle que soit la cible (= -marge). Les stratégies ne")
    print("    changent QUE la forme du risque, jamais l'espérance. Preuve chiffrée.")
