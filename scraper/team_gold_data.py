"""Données validées sur BDD CLEAN (3225 matchs, après cleanup 2026-06-05)."""

# ============ PAIRES OR HOME (parier 1) ============
PAIR_HOME_GOLD = {
    ("C. Palace", "Spurs"):             {"win": 0.875, "roi": 0.940, "cote": 2.22, "n": 8},
    ("N. Forest", "Everton"):           {"win": 1.000, "roi": 0.825, "cote": 1.83, "n": 8},
    ("N. Forest", "Newcastle"):         {"win": 0.667, "roi": 0.671, "cote": 2.50, "n": 9},
    ("C. Palace", "Bournemouth"):       {"win": 0.889, "roi": 0.666, "cote": 1.87, "n": 9},
    ("Brighton", "Manchester Red"):     {"win": 0.889, "roi": 0.596, "cote": 1.80, "n": 9},
    # ("Fulham", "London Blues") retiré 2026-06-05 : paire OUVERTE (3.09 buts/match, 27% nuls, 3-3 live)
    ("London Blues", "Liverpool"):      {"win": 0.667, "roi": 0.526, "cote": 2.27, "n": 12},
    ("Manchester Red", "Bournemouth"):  {"win": 0.909, "roi": 0.495, "cote": 1.64, "n": 11},
    ("A. Villa", "Fulham"):             {"win": 0.875, "roi": 0.465, "cote": 1.67, "n": 8},
    ("Spurs", "West Ham"):              {"win": 0.900, "roi": 0.461, "cote": 1.62, "n": 10},
    ("Fulham", "C. Palace"):            {"win": 0.700, "roi": 0.453, "cote": 2.08, "n": 10},
    ("N. Forest", "Bournemouth"):       {"win": 0.800, "roi": 0.450, "cote": 1.81, "n": 10},
    ("Brighton", "Bournemouth"):        {"win": 0.875, "roi": 0.440, "cote": 1.65, "n": 8},
    ("Liverpool", "London Reds"):       {"win": 0.600, "roi": 0.389, "cote": 2.31, "n": 10},
    ("N. Forest", "C. Palace"):         {"win": 0.667, "roi": 0.380, "cote": 2.07, "n": 12},
    ("Fulham", "Wolverhampton"):        {"win": 0.818, "roi": 0.372, "cote": 1.67, "n": 11},
    ("C. Palace", "West Ham"):          {"win": 0.750, "roi": 0.350, "cote": 1.79, "n": 8},
    ("London Blues", "A. Villa"):       {"win": 0.800, "roi": 0.336, "cote": 1.67, "n": 10},
    ("C. Palace", "Wolverhampton"):     {"win": 0.778, "roi": 0.322, "cote": 1.70, "n": 9},
    ("Bournemouth", "West Ham"):        {"win": 0.667, "roi": 0.310, "cote": 1.97, "n": 9},
    ("Brighton", "West Ham"):           {"win": 0.818, "roi": 0.306, "cote": 1.61, "n": 11},
    ("Newcastle", "Brighton"):          {"win": 0.889, "roi": 0.304, "cote": 1.47, "n": 9},
}

# ============ PAIRES OR AWAY (parier 2) ============
PAIR_AWAY_GOLD = {
    ("N. Forest", "Wolverhampton"):     {"win": 0.444, "roi": 1.206, "cote": 4.93, "n": 9, "max_cote_factor": 1.05},
    ("Fulham", "Brighton"):             {"win": 0.750, "roi": 1.035, "cote": 2.71, "n": 8, "max_cote_factor": 1.05},
    ("London Blues", "Manchester Red"): {"win": 0.417, "roi": 1.023, "cote": 4.88, "n": 12, "max_cote_factor": 1.05},
    ("London Blues", "Spurs"):          {"win": 0.500, "roi": 0.959, "cote": 3.88, "n": 8, "max_cote_factor": 1.05},
    ("Fulham", "Newcastle"):            {"win": 0.800, "roi": 0.936, "cote": 2.42, "n": 10, "max_cote_factor": 1.05},
    ("N. Forest", "Fulham"):            {"win": 0.556, "roi": 0.829, "cote": 3.25, "n": 9, "max_cote_factor": 1.05},
    ("Everton", "A. Villa"):            {"win": 0.583, "roi": 0.806, "cote": 3.12, "n": 12, "max_cote_factor": 1.05},
    ("C. Palace", "London Blues"):      {"win": 0.600, "roi": 0.804, "cote": 3.02, "n": 10, "max_cote_factor": 1.05},
    ("Brentford", "C. Palace"):         {"win": 0.500, "roi": 0.800, "cote": 3.63, "n": 10, "max_cote_factor": 1.05},
    ("C. Palace", "Manchester Red"):    {"win": 0.500, "roi": 0.718, "cote": 3.41, "n": 8, "max_cote_factor": 1.05},
    ("Spurs", "C. Palace"):             {"win": 0.417, "roi": 0.592, "cote": 3.81, "n": 12, "max_cote_factor": 1.05},
    ("Wolverhampton", "Brentford"):     {"win": 0.636, "roi": 0.583, "cote": 2.50, "n": 11, "max_cote_factor": 1.05},
    ("Bournemouth", "A. Villa"):        {"win": 0.500, "roi": 0.564, "cote": 3.16, "n": 8, "max_cote_factor": 1.05},
    ("C. Palace", "Newcastle"):         {"win": 0.625, "roi": 0.542, "cote": 2.47, "n": 8, "max_cote_factor": 1.05},
    ("Fulham", "Manchester Blue"):      {"win": 0.875, "roi": 0.510, "cote": 1.73, "n": 8, "max_cote_factor": 1.05},
}

# ============ PAIRES TRAP HOME (parier 1 perd) ============
PAIR_TRAP_HOME = {
    ("Everton", "London Blues"),
    ("London Blues", "Manchester Red"),
    ("Spurs", "N. Forest"),
    ("Spurs", "Burnley"),
}

# ============ BRACKETS OR HOME (recalculés sur BDD clean) ============
BRACKET_GOLD_HOME = {
    ("London Blues",  (2.1, 2.5)): 0.606,
    ("Burnley",       (2.5, 3.0)): 0.512,
    ("Brighton",      (1.9, 2.1)): 0.489,
    ("N. Forest",     (1.9, 2.1)): 0.461,
    ("N. Forest",     (1.7, 1.9)): 0.318,
    ("Bournemouth",   (1.5, 1.7)): 0.307,
    ("Manchester Red", (2.1, 2.5)): 0.301,
    ("Fulham",        (1.9, 2.1)): 0.292,
    ("Wolverhampton", (2.5, 3.0)): 0.273,
    ("C. Palace",     (1.7, 1.9)): 0.243,
    ("Brentford",     (1.3, 1.5)): 0.166,
    ("A. Villa",      (1.5, 1.7)): 0.162,
    ("Brighton",      (1.5, 1.7)): 0.153,
    ("Manchester Red", (1.5, 1.7)): 0.142,
    ("C. Palace",     (1.5, 1.7)): 0.130,
    ("West Ham",      (1.5, 1.7)): 0.109,
    ("London Blues",  (1.7, 1.9)): 0.106,
    ("Wolverhampton", (1.7, 1.9)): 0.101,
}

BRACKET_GOLD_AWAY = {
    ("Brentford",     (4.0, 100)): 0.340,
    ("London Blues",  (2.5, 3.0)): 0.336,
    ("Everton",       (1.7, 1.9)): 0.245,
    ("A. Villa",      (1.5, 1.7)): 0.237,
    ("A. Villa",      (4.0, 100)): 0.222,
    ("London Blues",  (3.0, 4.0)): 0.211,
    ("Manchester Red", (4.0, 100)): 0.207,
    ("Spurs",         (4.0, 100)): 0.189,
    ("Brentford",     (3.0, 4.0)): 0.168,
    ("A. Villa",      (3.0, 4.0)): 0.140,
    ("London Blues",  (1.9, 2.1)): 0.142,
    ("Wolverhampton", (4.0, 100)): 0.126,
    ("Brighton",      (2.1, 2.5)): 0.124,
    ("London Reds",   (1.9, 2.1)): 0.112,
    ("London Blues",  (4.0, 100)): 0.104,
    ("Bournemouth",   (1.5, 1.7)): 0.102,
}

BRACKET_TRAP_HOME = {
    ("Sunderland",    (3.0, 100)): -0.427,
    ("West Ham",      (1.9, 2.1)): -0.332,
    ("London Blues",  (3.0, 100)): -0.329,
    ("Bournemouth",   (3.0, 100)): -0.328,
    ("Wolverhampton", (2.5, 3.0)): -0.312,
    ("Everton",       (2.5, 3.0)): -0.304,
    ("Wolverhampton", (1.9, 2.1)): -0.295,
    ("Manchester Red",(3.0, 100)): -0.272,
    ("Everton",       (2.1, 2.5)): -0.258,
    ("N. Forest",     (1.5, 1.7)): -0.235,
    ("Spurs",         (1.9, 2.1)): -0.223,
    ("Bournemouth",   (1.7, 1.9)): -0.221,
}


def bracket_match(team, cote, table):
    for (t, (lo, hi)), roi in table.items():
        if t == team and lo <= cote < hi:
            return roi
    return None


# ============ OVER 2.5 GOLD (≥85% sur BDD clean) ============
OVER_GOLD = {
    ("A. Villa", "Fulham"): {"n": 8, "rate": 1.00},
    ("London Reds", "Brighton"): {"n": 9, "rate": 1.00},
    ("Manchester Red", "London Reds"): {"n": 12, "rate": 0.917},
    ("Sunderland", "Manchester Blue"): {"n": 11, "rate": 0.909},
    ("Fulham", "West Ham"): {"n": 11, "rate": 0.909},
    ("Newcastle", "West Ham"): {"n": 10, "rate": 0.90},
    ("Fulham", "Liverpool"): {"n": 10, "rate": 0.90},
    ("Fulham", "Manchester Red"): {"n": 10, "rate": 0.90},
    ("Manchester Blue", "Leeds"): {"n": 10, "rate": 0.90},
    ("Fulham", "Spurs"): {"n": 9, "rate": 0.889},
    ("Leeds", "Brighton"): {"n": 9, "rate": 0.889},
    ("Liverpool", "London Blues"): {"n": 9, "rate": 0.889},
    ("West Ham", "Bournemouth"): {"n": 9, "rate": 0.889},
    ("Spurs", "London Reds"): {"n": 9, "rate": 0.889},
    ("C. Palace", "Newcastle"): {"n": 8, "rate": 0.875},
    ("A. Villa", "Newcastle"): {"n": 8, "rate": 0.875},
    ("Fulham", "Manchester Blue"): {"n": 8, "rate": 0.875},
    ("Leeds", "Newcastle"): {"n": 8, "rate": 0.875},
    ("Liverpool", "A. Villa"): {"n": 8, "rate": 0.875},
    ("Manchester Blue", "Bournemouth"): {"n": 8, "rate": 0.875},
}

# ============ UNDER 2.5 GOLD (≤27% Over) ============
UNDER_GOLD = {
    ("Everton", "West Ham"): {"n": 11, "over_rate": 0.182},
    ("Manchester Blue", "C. Palace"): {"n": 14, "over_rate": 0.214},
    ("A. Villa", "C. Palace"): {"n": 9, "over_rate": 0.222},
    ("C. Palace", "N. Forest"): {"n": 12, "over_rate": 0.250},
    ("London Blues", "Manchester Red"): {"n": 12, "over_rate": 0.250},
    ("Burnley", "Wolverhampton"): {"n": 8, "over_rate": 0.250},
    ("Everton", "Brentford"): {"n": 8, "over_rate": 0.250},
    ("Sunderland", "London Blues"): {"n": 8, "over_rate": 0.250},
    ("Brighton", "Manchester Blue"): {"n": 8, "over_rate": 0.250},
    ("Bournemouth", "Manchester Blue"): {"n": 11, "over_rate": 0.273},
    ("Sunderland", "Brentford"): {"n": 11, "over_rate": 0.273},
}

# ============ BTTS OUI GOLD (≥80%) ============
BTTS_OUI_GOLD = {
    ("C. Palace", "Newcastle"): {"n": 8, "rate": 1.00, "min_cote_h": 1.5},
    ("A. Villa", "Fulham"): {"n": 8, "rate": 1.00, "min_cote_h": 1.6},
    ("Fulham", "Spurs"): {"n": 9, "rate": 1.00, "min_cote_h": 1.8},
    ("Leeds", "Burnley"): {"n": 9, "rate": 1.00, "min_cote_h": 1.8},
    ("A. Villa", "West Ham"): {"n": 11, "rate": 0.909, "min_cote_h": 1.8},
    ("West Ham", "Bournemouth"): {"n": 9, "rate": 0.889, "min_cote_h": 1.8},
    ("Manchester Blue", "Liverpool"): {"n": 9, "rate": 0.889, "min_cote_h": 1.5},
    ("Manchester Blue", "Newcastle"): {"n": 9, "rate": 0.889, "min_cote_h": 1.5},
    ("Spurs", "Burnley"): {"n": 9, "rate": 0.889, "min_cote_h": 1.6},
    ("Everton", "Fulham"): {"n": 8, "rate": 0.875, "min_cote_h": 1.8},
    ("Fulham", "Manchester Blue"): {"n": 8, "rate": 0.875, "min_cote_h": 1.8},
    ("Manchester Red", "N. Forest"): {"n": 12, "rate": 0.833, "min_cote_h": 1.8},
    ("N. Forest", "West Ham"): {"n": 11, "rate": 0.818, "min_cote_h": 1.8},
    ("A. Villa", "Manchester Blue"): {"n": 11, "rate": 0.818, "min_cote_h": 1.6},
    ("Bournemouth", "Brentford"): {"n": 10, "rate": 0.80, "min_cote_h": 1.8},
    ("N. Forest", "London Blues"): {"n": 10, "rate": 0.80, "min_cote_h": 1.8},
    ("London Reds", "West Ham"): {"n": 10, "rate": 0.80, "min_cote_h": 1.5},
    ("Fulham", "Manchester Red"): {"n": 10, "rate": 0.80, "min_cote_h": 1.8},
    ("Leeds", "A. Villa"): {"n": 10, "rate": 0.80, "min_cote_h": 1.8},
    ("Leeds", "Spurs"): {"n": 10, "rate": 0.80, "min_cote_h": 1.8},
}

# ============ BTTS NON GOLD (≤25%) ============
BTTS_NON_GOLD = {
    ("Liverpool", "Everton"): {"n": 11, "bts_rate": 0.091},
    ("Brighton", "Bournemouth"): {"n": 8, "bts_rate": 0.125},
    ("London Blues", "Manchester Red"): {"n": 12, "bts_rate": 0.167},
    ("N. Forest", "C. Palace"): {"n": 12, "bts_rate": 0.167},
    ("London Reds", "Leeds"): {"n": 12, "bts_rate": 0.167},
    ("Manchester Blue", "Leeds"): {"n": 10, "bts_rate": 0.20},
    ("Bournemouth", "N. Forest"): {"n": 9, "bts_rate": 0.222},
    ("Newcastle", "London Blues"): {"n": 9, "bts_rate": 0.222},
    ("Everton", "Brentford"): {"n": 8, "bts_rate": 0.25},
    ("N. Forest", "Everton"): {"n": 8, "bts_rate": 0.25},
    ("A. Villa", "Leeds"): {"n": 8, "bts_rate": 0.25},
    ("Sunderland", "London Blues"): {"n": 8, "bts_rate": 0.25},
    ("Sunderland", "Leeds"): {"n": 8, "bts_rate": 0.25},
    ("Manchester Blue", "A. Villa"): {"n": 8, "bts_rate": 0.25},
}

# ============ SCORE COMBO GOLD (top 2 ≥55% combo) ============
# Recalculé sur BDD clean
SCORE_COMBO_GOLD = {
    ("Fulham", "Manchester Blue"):    {"top1": "1-2", "r1": 0.625, "top2": "0-1", "r2": 0.125, "combo": 0.75, "n": 8},
    ("C. Palace", "Spurs"):           {"top1": "2-1", "r1": 0.50, "top2": "3-1", "r2": 0.125, "combo": 0.625, "n": 8},
    ("Burnley", "Manchester Blue"):   {"top1": "1-2", "r1": 0.50, "top2": "0-1", "r2": 0.083, "combo": 0.583, "n": 12},
    ("Newcastle", "West Ham"):        {"top1": "3-0", "r1": 0.50, "top2": "2-0", "r2": 0.10, "combo": 0.60, "n": 10},
    ("Wolverhampton", "Fulham"):      {"top1": "2-1", "r1": 0.50, "top2": "1-1", "r2": 0.20, "combo": 0.70, "n": 10},
    ("Bournemouth", "Manchester Blue"):{"top1": "1-1", "r1": 0.455, "top2": "0-3", "r2": 0.091, "combo": 0.545, "n": 11},
    ("Manchester Red", "Bournemouth"):{"top1": "1-0", "r1": 0.455, "top2": "3-1", "r2": 0.182, "combo": 0.636, "n": 11},
    ("Burnley", "Liverpool"):         {"top1": "0-2", "r1": 0.444, "top2": "0-1", "r2": 0.222, "combo": 0.667, "n": 9},
    ("A. Villa", "C. Palace"):        {"top1": "2-0", "r1": 0.444, "top2": "1-1", "r2": 0.111, "combo": 0.556, "n": 9},
    ("Leeds", "Brentford"):           {"top1": "1-2", "r1": 0.444, "top2": "0-3", "r2": 0.111, "combo": 0.556, "n": 9},
    ("Brighton", "Manchester Red"):   {"top1": "2-1", "r1": 0.444, "top2": "3-1", "r2": 0.222, "combo": 0.667, "n": 9},
    ("Manchester Red", "Wolverhampton"):{"top1": "1-0", "r1": 0.444, "top2": "1-2", "r2": 0.111, "combo": 0.556, "n": 9},
    ("West Ham", "A. Villa"):         {"top1": "1-1", "r1": 0.444, "top2": "2-0", "r2": 0.111, "combo": 0.556, "n": 9},
    ("Bournemouth", "Brentford"):     {"top1": "1-1", "r1": 0.40, "top2": "3-1", "r2": 0.20, "combo": 0.60, "n": 10},
    ("Liverpool", "London Reds"):     {"top1": "2-0", "r1": 0.40, "top2": "1-1", "r2": 0.10, "combo": 0.50, "n": 10},
    ("Manchester Blue", "Everton"):   {"top1": "3-0", "r1": 0.40, "top2": "2-1", "r2": 0.20, "combo": 0.60, "n": 10},
}


# ============ SCORE EXACT DOMINANT GOLD (≥37% sur BDD clean) ============
# Recalculé : sweet spot 30-45%, éviter ≥50% (over-fit prouvé)
SCORE_DOMINANT_GOLD = {
    ("Fulham", "Manchester Blue"): {"score": "1-2", "rate": 0.625, "n": 8},
    ("C. Palace", "Spurs"): {"score": "2-1", "rate": 0.50, "n": 8},
    ("Burnley", "Manchester Blue"): {"score": "1-2", "rate": 0.50, "n": 12},
    ("Newcastle", "West Ham"): {"score": "3-0", "rate": 0.50, "n": 10},
    ("Wolverhampton", "Fulham"): {"score": "2-1", "rate": 0.50, "n": 10},
    ("Bournemouth", "Manchester Blue"): {"score": "1-1", "rate": 0.455, "n": 11},
    ("Manchester Red", "Bournemouth"): {"score": "1-0", "rate": 0.455, "n": 11},
    ("Burnley", "Liverpool"): {"score": "0-2", "rate": 0.444, "n": 9},
    ("A. Villa", "C. Palace"): {"score": "2-0", "rate": 0.444, "n": 9},
    ("Leeds", "Brentford"): {"score": "1-2", "rate": 0.444, "n": 9},
    ("Brighton", "Manchester Red"): {"score": "2-1", "rate": 0.444, "n": 9},
    ("Manchester Red", "Wolverhampton"): {"score": "1-0", "rate": 0.444, "n": 9},
    ("West Ham", "A. Villa"): {"score": "1-1", "rate": 0.444, "n": 9},
    ("Bournemouth", "Brentford"): {"score": "1-1", "rate": 0.40, "n": 10},
    ("Liverpool", "London Reds"): {"score": "2-0", "rate": 0.40, "n": 10},
    ("Manchester Blue", "Everton"): {"score": "3-0", "rate": 0.40, "n": 10},
    ("C. Palace", "Burnley"): {"score": "4-0", "rate": 0.375, "n": 8},
    ("C. Palace", "Manchester Red"): {"score": "1-2", "rate": 0.375, "n": 8},
    ("Burnley", "Wolverhampton"): {"score": "1-0", "rate": 0.375, "n": 8},
    ("Everton", "Bournemouth"): {"score": "1-1", "rate": 0.375, "n": 8},
    ("Bournemouth", "Everton"): {"score": "1-1", "rate": 0.375, "n": 8},
    ("N. Forest", "Everton"): {"score": "3-0", "rate": 0.375, "n": 8},
    ("A. Villa", "Liverpool"): {"score": "1-2", "rate": 0.375, "n": 8},
    ("A. Villa", "Fulham"): {"score": "3-1", "rate": 0.375, "n": 8},
    ("Newcastle", "Manchester Blue"): {"score": "0-0", "rate": 0.375, "n": 8},
}
