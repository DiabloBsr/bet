"""JOURNAL DE BANKROLL — paris réels, courbe, ROI, stop-loss / take-profit.

Base dédiée data/bankroll.db (évite les locks du scraper sur la base principale).
render(st) : UI complète à embarquer dans le dashboard.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "bankroll.db"

DDL = [
    """CREATE TABLE IF NOT EXISTS bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, match TEXT, market TEXT,
        sel TEXT, stake REAL, odds REAL, status TEXT DEFAULT 'pending', pnl REAL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS cfg (k TEXT PRIMARY KEY, v REAL)""",
]


def _con():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    for q in DDL:
        c.execute(q)
    return c


def get_cfg(key, default):
    c = _con()
    r = c.execute("SELECT v FROM cfg WHERE k=?", (key,)).fetchone()
    c.close()
    return r[0] if r else default


def set_cfg(key, val):
    c = _con(); c.execute("INSERT INTO cfg(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?",
                          (key, val, val)); c.commit(); c.close()


def add_bet(match, market, sel, stake, odds):
    c = _con()
    c.execute("INSERT INTO bets(created_at,match,market,sel,stake,odds) VALUES(?,?,?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(), match, market, sel, float(stake), float(odds)))
    c.commit(); c.close()


def settle(bet_id, status):
    c = _con()
    row = c.execute("SELECT stake,odds FROM bets WHERE id=?", (bet_id,)).fetchone()
    if row:
        stake, odds = row
        pnl = stake*(odds-1) if status == "win" else (-stake if status == "loss" else 0.0)
        c.execute("UPDATE bets SET status=?, pnl=? WHERE id=?", (status, pnl, bet_id))
        c.commit()
    c.close()


def delete_bet(bet_id):
    c = _con(); c.execute("DELETE FROM bets WHERE id=?", (bet_id,)); c.commit(); c.close()


def render(st):
    import pandas as pd
    st.subheader("💰 Mon bankroll — journal & discipline")
    start = get_cfg("start", 100000.0)
    sl = get_cfg("stop_loss", 20.0)      # % de perte de session
    tp = get_cfg("take_profit", 30.0)    # % de gain de session

    c1, c2, c3 = st.columns(3)
    new_start = c1.number_input("Bankroll de départ", 1000.0, 1e8, float(start), 1000.0)
    new_sl = c2.number_input("Stop-loss session (%)", 5.0, 90.0, float(sl), 5.0)
    new_tp = c3.number_input("Take-profit session (%)", 5.0, 200.0, float(tp), 5.0)
    if (new_start, new_sl, new_tp) != (start, sl, tp):
        set_cfg("start", new_start); set_cfg("stop_loss", new_sl); set_cfg("take_profit", new_tp)

    df = pd.read_sql("SELECT * FROM bets ORDER BY id", _con())
    settled = df[df.status != "pending"]
    pnl = float(settled.pnl.sum()) if len(settled) else 0.0
    bankroll = new_start + pnl
    staked = float(settled.stake.sum()) if len(settled) else 0.0
    roi = 100*pnl/staked if staked else 0.0
    wins = int((settled.status == "win").sum()); nres = len(settled)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bankroll actuelle", f"{bankroll:,.0f}", f"{pnl:+,.0f}")
    k2.metric("ROI réel", f"{roi:+.1f}%", f"{nres} paris réglés")
    k3.metric("Réussite", f"{100*wins/nres:.0f}%" if nres else "—")
    k4.metric("Total misé", f"{staked:,.0f}")

    # stop-loss / take-profit
    change = 100*pnl/new_start if new_start else 0.0
    if change <= -new_sl:
        st.error(f"🛑 STOP-LOSS ATTEINT ({change:+.1f}% ≤ −{new_sl:.0f}%) — ARRÊTE de jouer aujourd'hui. "
                 "C'est la règle qui protège ta bankroll. Reviens demain à froid.")
    elif change >= new_tp:
        st.success(f"🎯 TAKE-PROFIT ATTEINT ({change:+.1f}% ≥ +{new_tp:.0f}%) — encaisse et ARRÊTE. "
                   "Ne rends pas tes gains au book.")
    else:
        st.caption(f"Session : {change:+.1f}% | stop-loss à −{new_sl:.0f}% · take-profit à +{new_tp:.0f}%")

    # courbe de bankroll
    if len(settled):
        cur = new_start + settled.pnl.cumsum()
        st.line_chart(pd.DataFrame({"bankroll": [new_start] + list(cur)}))

    # ---- ajouter un pari ----
    with st.expander("➕ Enregistrer un pari"):
        a1, a2 = st.columns([3, 2])
        mt = a1.text_input("Match / description", key="bk_m")
        mk = a2.text_input("Marché", value="1X2", key="bk_mk")
        b1, b2, b3 = st.columns(3)
        sel = b1.text_input("Sélection", value="1", key="bk_s")
        stake = b2.number_input("Mise", 1.0, 1e7, 4000.0, 100.0, key="bk_st")
        odds = b3.number_input("Cote", 1.01, 1000.0, 2.0, 0.05, key="bk_o")
        if st.button("Ajouter", key="bk_add") and mt:
            add_bet(mt, mk, sel, stake, odds); st.rerun()

    # ---- liste + règlement ----
    if len(df):
        st.markdown("**Paris**")
        for r in df.iloc[::-1].head(25).itertuples():
            cc = st.columns([4, 2, 2, 3])
            badge = "🟢" if r.status == "win" else ("🔴" if r.status == "loss" else "⏳")
            cc[0].markdown(f"{badge} **{r.match[:32]}** — {r.sel} `[{r.market}]`")
            cc[1].caption(f"{r.stake:g} @ {r.odds:g}")
            cc[2].caption(f"{r.pnl:+,.0f}" if r.status != "pending" else "en attente")
            if r.status == "pending":
                x = cc[3].columns(3)
                if x[0].button("✅", key=f"w{r.id}"): settle(r.id, "win"); st.rerun()
                if x[1].button("❌", key=f"l{r.id}"): settle(r.id, "loss"); st.rerun()
                if x[2].button("🗑", key=f"d{r.id}"): delete_bet(r.id); st.rerun()
            else:
                if cc[3].button("↩ annuler", key=f"u{r.id}"): settle(r.id, "pending"); st.rerun()
    st.caption("⚠️ Aucun pari n'a d'espérance positive (RNG calibré). La bankroll ne se protège "
               "PAS en gagnant plus, mais en misant petit et en respectant ton stop-loss. "
               "C'est le seul 'edge' réel : la discipline.")
