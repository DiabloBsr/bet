"""Thème UI moderne pour les dashboards Streamlit — inspiré Magic UI (dégradé animé,
shimmer, glow, glassmorphism) répliqué en CSS pur (Streamlit ne peut pas embarquer React).

Usage :
    from scripts.ui_theme import inject_theme, hero
    inject_theme(st, accent="#22c55e")
    hero(st, "⚖️ Prédiction TRIO", "V2 + V5 + arbitre Marché", badges=[...])
"""
from __future__ import annotations


def inject_theme(st, accent: str = "#22c55e", accent2: str = "#2dd4bf", accent3: str = "#38bdf8"):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {{
  --accent: {accent}; --accent2: {accent2}; --accent3: {accent3};
  --bg: #07090c; --surface: rgba(255,255,255,.035); --stroke: rgba(255,255,255,.09);
  --text: #e8eef4; --muted: #8b98a6;
}}

/* fond near-black + halos + grille de points (dot-pattern glow) */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1100px 550px at 12% -8%, rgba(34,197,94,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(56,189,248,.12), transparent 55%),
    radial-gradient(circle at 50% 120%, rgba(45,212,191,.08), transparent 50%),
    var(--bg);
}}
[data-testid="stAppViewContainer"]::before {{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
  background-size: 22px 22px; mask-image: radial-gradient(1200px 600px at 50% 0%, #000 25%, transparent 75%);
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; position: relative; z-index: 1; max-width: 1300px; }}
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; color: var(--text); }}
h1,h2,h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: -.02em; }}

/* ---- HERO ---- */
.trio-hero {{ margin: 0 0 1.4rem; }}
.trio-title {{
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.5rem; line-height:1.05; margin:0;
  background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3), var(--accent));
  background-size: 300% 100%; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation: trioGrad 6s linear infinite; filter: drop-shadow(0 2px 18px rgba(34,197,94,.25));
}}
@keyframes trioGrad {{ to {{ background-position: 300% 0; }} }}
.trio-sub {{
  margin:.55rem 0 0; font-size:1rem; font-weight:500;
  background: linear-gradient(90deg,var(--muted) 0%,var(--muted) 42%,#fff 50%,var(--muted) 58%,var(--muted) 100%);
  background-size:200% 100%; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation: trioShine 6s linear infinite;
}}
@keyframes trioShine {{ to {{ background-position:-200% 0; }} }}
.trio-badges {{ margin-top:.85rem; display:flex; gap:.5rem; flex-wrap:wrap; }}
.trio-badge {{
  font-size:.76rem; font-weight:600; padding:.32rem .7rem; border-radius:999px;
  background: var(--surface); border:1px solid var(--stroke); color:var(--text);
  backdrop-filter: blur(8px); box-shadow: inset 0 0 12px rgba(34,197,94,.05);
}}
.trio-badge b {{ color: var(--accent2); }}

/* ---- cartes métriques glassmorphism ---- */
[data-testid="stMetric"] {{
  background: var(--surface); border:1px solid var(--stroke); border-radius:16px;
  padding:1rem 1.1rem; backdrop-filter: blur(10px);
  box-shadow: 0 8px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-3px); border-color: rgba(34,197,94,.4);
  box-shadow: 0 14px 40px rgba(0,0,0,.45), 0 0 22px rgba(34,197,94,.15);
}}
[data-testid="stMetricValue"] {{ font-family:'Space Grotesk',sans-serif; font-weight:700; color:#fff; }}
[data-testid="stMetricLabel"] p {{ color: var(--muted); font-weight:500; }}

/* ---- boutons ---- */
.stButton > button, .stDownloadButton > button {{
  border:1px solid rgba(34,197,94,.35); border-radius:12px; font-weight:600;
  background: linear-gradient(180deg, rgba(34,197,94,.22), rgba(34,197,94,.08));
  color:#eafff2; transition: all .16s ease; box-shadow: 0 0 0 rgba(34,197,94,0);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--accent); transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(34,197,94,.28), 0 0 18px rgba(34,197,94,.2);
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg, var(--accent), var(--accent2)); color:#04140b; border:none;
}}

/* ---- onglets pilules ---- */
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; background:transparent; border-bottom:1px solid var(--stroke); }}
.stTabs [data-baseweb="tab"] {{
  background: var(--surface); border:1px solid var(--stroke); border-radius:11px 11px 4px 4px;
  padding:.4rem .9rem; color:var(--muted); font-weight:600;
}}
.stTabs [aria-selected="true"] {{
  color:#fff; border-color: rgba(34,197,94,.5);
  background: linear-gradient(180deg, rgba(34,197,94,.2), rgba(34,197,94,.04));
  box-shadow: 0 0 18px rgba(34,197,94,.15);
}}

/* ---- expanders / conteneurs ---- */
[data-testid="stExpander"] {{
  background: var(--surface); border:1px solid var(--stroke); border-radius:14px; backdrop-filter: blur(8px);
}}
[data-testid="stExpander"] summary:hover {{ color: var(--accent2); }}
[data-testid="stAlert"], [data-testid="stNotification"] {{ border-radius:14px; border:1px solid var(--stroke); }}
hr {{ border:none; height:1px; background:linear-gradient(90deg,transparent,var(--stroke),transparent); }}

/* inputs */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
  background: var(--surface) !important; border:1px solid var(--stroke) !important; border-radius:11px !important;
}}
/* scrollbar */
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-thumb {{ background: rgba(34,197,94,.35); border-radius:10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)


def hero(st, title: str, subtitle: str = "", badges=None):
    b = ""
    if badges:
        b = '<div class="trio-badges">' + "".join(f'<span class="trio-badge">{x}</span>' for x in badges) + "</div>"
    st.markdown(
        f'<div class="trio-hero"><h1 class="trio-title">{title}</h1>'
        f'<p class="trio-sub">{subtitle}</p>{b}</div>', unsafe_allow_html=True)
