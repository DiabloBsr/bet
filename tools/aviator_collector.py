"""COLLECTEUR Aviator — session connectée + capture ROUNDS_HISTORY -> SQLite.

Stocke chaque manche {round_id (unique), multiplier, captured_at} dans data/aviator.db.
Dédup par round_id (comme content_hash du foot). Re-login si la session tombe.

Usage : BET_USER=.. BET_PASS=.. python tools/aviator_collector.py [minutes]
        minutes=0 -> tourne en continu (jusqu'à Ctrl-C / fermeture).
"""
from __future__ import annotations
import os, re, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path
import msgpack
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "aviator.db"
SESSION = ROOT / ".bet_session"
BET_URL = "https://bet261.mg/instant-games/llc/Aviator"
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 3
USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")

DB.parent.mkdir(parents=True, exist_ok=True)
db = sqlite3.connect(DB)
db.execute("""CREATE TABLE IF NOT EXISTS aviator_rounds (
    round_id TEXT PRIMARY KEY, multiplier REAL, captured_at TEXT)""")
db.commit()

seen = set(r[0] for r in db.execute("SELECT round_id FROM aviator_rounds"))
_new = [0]


def store(rounds):
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in rounds:
        rid = r.get("_id"); mult = r.get("multiplierCrash")
        if rid and rid not in seen and isinstance(mult, (int, float)):
            seen.add(rid); rows.append((rid, float(mult), now))
    if rows:
        db.executemany("INSERT OR IGNORE INTO aviator_rounds VALUES (?,?,?)", rows)
        db.commit(); _new[0] += len(rows)
    return len(rows)


def decode(raw):
    if isinstance(raw, str):
        raw = raw.encode("latin-1", "ignore")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    for start in (0, 1):
        try:
            obj = msgpack.unpackb(raw[start:], raw=False, strict_map_key=False)
            if isinstance(obj, dict) and isinstance(obj.get("data"), list) and obj["data"]:
                d = obj["data"]
                return str(d[0]), (d[1] if len(d) > 1 else None)
        except Exception:
            continue
    return None


def click_ok(page):
    for pat in ("OK", "Oui", "Accepter"):
        try:
            el = page.get_by_role("button", name=re.compile(f"^{pat}$", re.I)).first
            if el.is_visible(timeout=1200):
                el.click(timeout=1200); return
        except Exception:
            continue


def login(page):
    try:
        if page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=2000) and USER:
            click_ok(page)
            page.locator('#Login_Id, input[name="username"]').first.fill(USER, timeout=2500)
            page.locator('#Login_Password, input[type="password"]').first.fill(PWD, timeout=2500)
            page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.click(timeout=2500)
            time.sleep(10)
    except Exception:
        pass


def main():
    deadline = time.time() + MINUTES * 60 if MINUTES > 0 else float("inf")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(SESSION), headless=False, args=["--disable-blink-features=AutomationControlled"],
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            viewport={"width": 1300, "height": 850})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded"); time.sleep(4)
        login(page)
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded"); time.sleep(3)
        for _ in range(3):
            click_ok(page); time.sleep(2)
        game_url = next((f.url for f in page.frames if f.url and "aviator.studio" in f.url), None)
        if not game_url:
            print("⚠ jeu introuvable (connexion ?) — abandon"); ctx.close(); return

        def on_ws(ws):
            def handle(pl):
                r = decode(pl)
                if not r:
                    return
                ev, payload = r
                if ev == "ROUNDS_HISTORY" and isinstance(payload, dict):
                    n = store(payload.get("rounds", []))
                    if n:
                        print(f"  +{n} manches (total {len(seen)})", flush=True)
                elif ev == "ONGOING_ROUND" and isinstance(payload, dict):
                    prev = (payload.get("previousRoundData") or {}).get("multiplier")
                    rid = payload.get("roundId")
                    # la manche précédente n'a pas toujours d'_id ici -> ignorée, ROUNDS_HISTORY couvre
            ws.on("framereceived", handle)
        page.on("websocket", on_ws)

        page.goto(game_url, timeout=45000, wait_until="domcontentloaded"); time.sleep(3)
        click_ok(page)
        print(f"COLLECTE démarrée (DB={DB.name}, {len(seen)} manches déjà en base)…", flush=True)
        last = time.time()
        while time.time() < deadline:
            time.sleep(5)
            if time.time() - last > 60:
                print(f"  … {len(seen)} manches en base (+{_new[0]} cette session)", flush=True)
                last = time.time()
            # re-nav si la page a été fermée / le socket perdu
            try:
                if not any("aviator.studio" in (f.url or "") for f in page.frames):
                    page.goto(game_url, timeout=30000); time.sleep(3); click_ok(page)
            except Exception:
                break
        ctx.close()

    tot = db.execute("SELECT COUNT(*) FROM aviator_rounds").fetchone()[0]
    print(f"\nCOLLECTE terminée : {tot} manches en base (+{_new[0]} cette session) -> {DB}")


if __name__ == "__main__":
    main()
