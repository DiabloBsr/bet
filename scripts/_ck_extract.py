"""Track B - extraction des features 'cles combinees' vers un artefact compact.

Une ligne par event (snapshot OUVERTURE) : signaux derives du vecteur de marche
+ resultat + COTES OFFERTES (Total / Score exact) du resultat realise (pour l'EV).
Anti-MemoryError : lecture chunkee, parse JSON ligne par ligne puis discard, on
n'accumule QUE ~30 floats par event (jamais le JSON brut).

Sortie : exports/combokeys_features.parquet (+ .csv fallback)
         exports/combokeys_binspec.json (bornes de bins gelees, partagees live)

Usage: ./.venv/Scripts/python.exe scripts/_ck_extract.py [--all-leagues]
"""
import sys
sys.path.insert(0, ".")
import argparse
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
from scraper.market_inversion import (
    invert_markets, parse_extra_markets, devig_market, total_buts_odds,
    score_exact_odds, _get_market,
)

CAP = 6
CHUNK = 2000
# scores frequents dont on stocke la cote offerte (pour l'EV de n'importe quelle prediction)
SCORES = ["1-1", "2-1", "1-2", "1-0", "0-1", "2-0", "0-2", "0-0", "2-2",
          "3-0", "0-3", "3-1", "1-3", "3-2", "2-3"]

ap = argparse.ArgumentParser()
ap.add_argument("--all-leagues", action="store_true")
args = ap.parse_args()

e = create_engine(load_settings().db_url)
corrupted = load_corrupted_ids()

comp_filter = "" if args.all_leagues else "AND e.competition='InstantLeague-8035'"
SQL = f"""
    SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start,
           o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.expected_start IS NOT NULL
      AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
      {comp_filter}
"""


def _safe_cote(v):
    try:
        f = float(v)
        return f if 1.0 < f < 99.99 else None
    except (TypeError, ValueError):
        return None


def row_features(row) -> dict | None:
    oh, od, oa = row.oh, row.od, row.oa
    if not (oh and od and oa) or oh <= 1 or oa <= 1 or od <= 1:
        return None
    em = parse_extra_markets(row.extra_markets)
    r = invert_markets(oh, od, oa, em)
    lh, la = r.lam_h, r.lam_a
    fav, dog = min(oh, oa), max(oh, oa)
    sa, sb = int(row.sa), int(row.sb)
    total = min(sa + sb, CAP)
    score = f"{sa}-{sb}"

    # Double Chance implicite (devig target_sum=2)
    dc = _get_market(em, exact="Double Chance")
    dc_d = {}
    if isinstance(dc, dict):
        dc_d = devig_market({k: dc.get(k) for k in ("1X", "X2", "12") if dc.get(k)}, target_sum=2.0)

    # Total de buts implicite
    tb = total_buts_odds(em)
    tb_d = devig_market(tb) if len(tb) >= 4 else {}
    pt = {k: tb_d.get(str(k), np.nan) for k in range(CAP + 1)}

    # G/NG implicite
    gng = _get_market(em, exact="G/NG")
    p_btts = np.nan
    if isinstance(gng, dict) and gng.get("Oui") and gng.get("Non"):
        d = devig_market({"Oui": gng["Oui"], "Non": gng["Non"]})
        p_btts = d.get("Oui", np.nan)

    # cotes offertes : vecteurs complets (pour l'EV de N'IMPORTE QUELLE prediction)
    se = score_exact_odds(em)
    off_total = tb.get(str(total))
    off_score = se.get(score)
    dc_x2_cote = dc.get("X2") if isinstance(dc, dict) else None
    # cote Over/Under 3.5 (marche '+/-', seul O/U cumulatif offert)
    ou = _get_market(em, exact="+/-")
    ou_o35 = ou_u35 = np.nan
    if isinstance(ou, dict):
        ous = {str(kk).replace(" ", ""): vv for kk, vv in ou.items()}
        o = _safe_cote(ous.get(">3.5")); u = _safe_cote(ous.get("<3.5"))
        ou_o35 = o if o else np.nan
        ou_u35 = u if u else np.nan
    off_t = {f"off_t{k}": (round(tb[str(k)], 3) if tb.get(str(k)) else np.nan) for k in range(CAP + 1)}
    off_s = {f"off_s_{s}": (round(se[s], 3) if se.get(s) else np.nan) for s in SCORES}

    return dict(
        id=int(row.id), competition=row.competition, team_a=row.team_a, team_b=row.team_b,
        expected_start=row.expected_start,
        oh=round(oh, 3), od=round(od, 3), oa=round(oa, 3),
        fav=round(fav, 3), dog=round(dog, 3), odds_ratio=round(dog / fav, 3),
        lam_h=round(lh, 4), lam_a=round(la, 4), lam_tot=round(lh + la, 4), lam_diff=round(lh - la, 4),
        dc_1X=dc_d.get("1X", np.nan), dc_X2=dc_d.get("X2", np.nan), dc_12=dc_d.get("12", np.nan),
        dc_x2_cote=round(dc_x2_cote, 3) if dc_x2_cote else np.nan,
        p_total_le2=round(pt[0] + pt[1] + pt[2], 4) if not any(np.isnan([pt[0], pt[1], pt[2]])) else np.nan,
        p_total_eq3=round(pt[3], 4) if not np.isnan(pt[3]) else np.nan,
        p_total_ge4=round(pt[4] + pt[5] + pt[6], 4) if not any(np.isnan([pt[4], pt[5], pt[6]])) else np.nan,
        p_btts=round(p_btts, 4) if not np.isnan(p_btts) else np.nan,
        residual=r.residual, gng_gap=r.per_market_gap.get("gng", np.nan),
        total_gap=r.per_market_gap.get("total", np.nan), score_gap=r.per_market_gap.get("score_exact", np.nan),
        fit_quality=r.fit_quality,
        total_goals=total, exact_score=score,
        off_total_cote=round(off_total, 3) if off_total else np.nan,
        off_score_cote=round(off_score, 3) if off_score else np.nan,
        off_ou_over35=round(ou_o35, 3) if ou_o35 == ou_o35 else np.nan,
        off_ou_under35=round(ou_u35, 3) if ou_u35 == ou_u35 else np.nan,
        **off_t, **off_s,
    )


records = []
n_seen = 0
for chunk in pd.read_sql(SQL, e, chunksize=CHUNK):
    chunk = chunk[~chunk.id.isin(corrupted)]
    for row in chunk.itertuples():
        n_seen += 1
        feat = row_features(row)
        if feat is not None:
            records.append(feat)
    print(f"  ...{n_seen} events vus, {len(records)} retenus", flush=True)

df = pd.DataFrame(records)
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
# dedup IDENTITE D'EVENT uniquement (jamais sur le score)
df = df.drop_duplicates(["team_a", "team_b", "expected_start"], keep="first").reset_index(drop=True)
print(f"\nfinal: {len(df)} events  | leagues={'ALL' if args.all_leagues else '8035'}")

stem = "combokeys_features_all" if args.all_leagues else "combokeys_features"
out_parquet = f"exports/{stem}.parquet"
try:
    df.to_parquet(out_parquet, index=False)
    print(f"ecrit {out_parquet}")
except Exception as ex:
    out_csv = f"exports/{stem}.csv"
    df.to_csv(out_csv, index=False)
    print(f"parquet KO ({ex}); ecrit {out_csv}")

# --- bornes de bins gelees (partagees avec le predictor live) ---
binspec = {
    "fav": [1.0, 1.35, 1.70, 2.10, 2.40, 2.70, 99],
    "dog": [1.0, 1.70, 2.10, 2.40, 2.70, 3.20, 99],
    "odds_ratio": [1.0, 1.15, 1.4, 1.8, 2.6, 99],
    "od": [1.0, 3.6, 4.2, 4.8, 99],
    "dc_x2_cote": [1.0, 1.25, 1.35, 1.6, 2.2, 99],
    "p_total_eq3": [0, 0.20, 0.24, 0.28, 1.0],
    "p_total_le2": [0, 0.25, 0.40, 0.55, 1.0],
    "p_btts": [0, 0.45, 0.55, 0.65, 1.0],
    "lam_tot": [0, 2.4, 2.8, 3.2, 99],
    "lam_diff": [-9, -0.3, 0.3, 9],
    "residual": [0, 0.03, 0.045, 0.06, 1.0],
}
with open("exports/combokeys_binspec.json", "w", encoding="utf-8") as f:
    json.dump(binspec, f, indent=1)
print("ecrit exports/combokeys_binspec.json")

# apercu coverage des marches
for c in ["dc_X2", "p_total_eq3", "p_btts", "off_total_cote", "off_score_cote"]:
    cov = 100 * df[c].notna().mean()
    print(f"  coverage {c:<16}: {cov:.0f}%")
