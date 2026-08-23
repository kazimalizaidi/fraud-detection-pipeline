"""
AegisWall — Financial Fraud & Anomaly Detection Dashboard
Restyled onto the "Orbit" dark, orange-accented fintech UI system.
All streaming / scoring / SHAP logic preserved exactly from the previous UI.

Run with: streamlit run app.py
(Ship the accompanying .streamlit/config.toml alongside this file — it sets
the app's primaryColor to orange so sliders/buttons/etc. use the theme
instead of Streamlit's default purple.)
"""

import joblib
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AegisWall · Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="locked",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS  (Orbit system)
# ----------------------------------------------------------------------------
BG          = "#0A0A0C"
SURFACE     = "#151517"
SURFACE_ALT = "#1C1C1F"
BORDER      = "#2A2A2E"
ORANGE      = "#FF5A1F"
ORANGE_SOFT = "#FF8A50"
TEXT        = "#F2F2F4"
SUBTEXT     = "#8B8B93"
GREEN       = "#3ED598"
RED         = "#F76A6A"
BLUE        = "#5B8DEF"
YELLOW      = "#F5B84F"

RISK_THRESHOLD = 0.5
STREAM_INTERLEAVE_GAP = 150  # inserts one fraud case roughly every N legit transactions

# ----------------------------------------------------------------------------
# ICONS (inline SVG — stroke based, no emoji, matches Orbit reference)
# ----------------------------------------------------------------------------
ICONS = {
    "grid": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "activity": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "target": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg>',
    "alert": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "shield": '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 3 6v6c0 5.25 3.75 9.75 9 11 5.25-1.25 9-5.75 9-11V6z"/></svg>',
    "shield-line": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/></svg>',
    "file": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "sliders": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "bolt": '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 3 14h7l-1 8 11-14h-7z"/></svg>',
    "search": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "bell": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>',
    "refresh": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
    "up": '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
    "dots": '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>',
    "eye": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "database": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
}


def icon(name, color="currentColor"):
    return ICONS.get(name, "").replace("currentColor", color)


# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Keep Streamlit's header mounted. The sidebar is locked open via
   st.set_page_config(), so we do not need fragile DOM selectors or JS
   workarounds for the collapse/reopen control. */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}

header {{
    background: transparent !important;
}}

/* Sidebar is intentionally locked open. Streamlit 1.61+ handles this natively. */

.stApp {{
    background: {BG};
    color: {TEXT};
}}

.block-container {{
    padding: 1.6rem 2.4rem 3rem 2.4rem;
    max-width: 1500px;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding: 1.6rem 1.1rem;
}}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ---------- Typography helpers ---------- */
.eyebrow {{
    font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
    color: {SUBTEXT}; font-weight: 600; margin-bottom: 6px;
}}
.display-num {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; letter-spacing: -0.02em;
}}

/* ---------- Brand ---------- */
.brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 4px 4px 16px 4px;
}}
.brand-mark {{
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(140deg, {ORANGE}, {ORANGE_SOFT});
    display: flex; align-items: center; justify-content: center;
    color: #100701; box-shadow: 0 4px 14px rgba(255,90,31,.35);
    flex-shrink: 0;
}}
.brand-name {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 18px; color: {TEXT}; letter-spacing: -0.01em;
}}
.brand-sub {{ font-size: 11px; color: {SUBTEXT}; margin-top: -2px; }}

/* ---------- Nav (real st.button-based, restyled) ---------- */
.nav-section-label {{
    font-size: 10.5px; letter-spacing: .12em; color: #55555c;
    font-weight: 700; margin: 16px 6px 8px 6px;
}}
.navitem {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 9px 12px; border-radius: 10px;
    color: {SUBTEXT}; font-size: 13.5px; font-weight: 500;
    margin-bottom: 2px;
}}
.navbadge {{
    font-size: 10.5px; font-weight: 700; padding: 1px 7px; border-radius: 20px;
    background: {SURFACE_ALT}; color: {SUBTEXT}; border: 1px solid {BORDER};
}}
.navbadge.live {{ background: rgba(62,213,152,0.15); color: {GREEN}; border: none; }}

/* Sidebar nav buttons: full width, left aligned, flat by default.
   st.container(key="nav_block") adds a real "st-key-nav_block" class to
   its wrapping element, so these selectors actually match (unlike the
   split-<div> approach used previously). */
section[data-testid="stSidebar"] .st-key-nav_block .stButton button {{
    width: 100%;
    justify-content: flex-start !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 12px !important;
    margin-bottom: 3px !important;
    gap: 10px !important;
}}
/* Streamlit nests the icon+label inside an inner wrapper div that has its
   own centering — the outer button's justify-content alone isn't enough,
   so we target every descendant flex container too. */
section[data-testid="stSidebar"] .st-key-nav_block .stButton button > div,
section[data-testid="stSidebar"] .st-key-nav_block .stButton button div {{
    justify-content: flex-start !important;
    width: 100% !important;
    text-align: left !important;
}}
section[data-testid="stSidebar"] .st-key-nav_block .stButton button p {{
    text-align: left !important;
}}
/* Streamlit nests the icon+label inside inner wrapper elements that default
   to centered content — flex-start on the outer <button> alone isn't enough,
   so we also target every inner div/p/span that could be centering things. */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .stButton button span {{
    justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] .stButton button p {{
    text-align: left !important;
}}
section[data-testid="stSidebar"] .st-key-nav_block button[kind="secondary"] {{
    background: transparent !important;
    color: {SUBTEXT} !important;
    border: 1px solid transparent !important;
}}
section[data-testid="stSidebar"] .st-key-nav_block button[kind="secondary"]:hover {{
    background: {SURFACE_ALT} !important;
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] .st-key-nav_block button[kind="secondary"] span[data-testid="stIconMaterial"] {{
    color: {SUBTEXT} !important;
}}
/* Active nav button: soft orange pill, matching the original navitem.active look */
section[data-testid="stSidebar"] .st-key-nav_block button[kind="primary"] {{
    background: rgba(255,90,31,0.14) !important;
    color: {TEXT} !important;
    border: 1px solid rgba(255,90,31,0.35) !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .st-key-nav_block button[kind="primary"] span[data-testid="stIconMaterial"] {{
    color: {ORANGE} !important;
}}

/* ---------- Stream control card (sidebar) ---------- */
.stream-card {{
    margin-top: 22px; padding: 16px; border-radius: 14px;
    background: {SURFACE_ALT}; border: 1px solid {BORDER};
}}
.stream-card .t {{
    font-weight: 700; font-size: 13px; color: {TEXT};
    display: flex; align-items: center; gap: 8px; margin-bottom: 2px;
}}
.stream-card .d {{ font-size: 11.5px; color: {SUBTEXT}; line-height: 1.4; margin-bottom: 10px; }}
.status-dot {{
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px;
}}
.status-dot.on {{ background: {GREEN}; box-shadow: 0 0 8px {GREEN}; }}
.status-dot.off {{ background: {SUBTEXT}; }}
.status-line {{
    font-size: 12px; color: {SUBTEXT}; font-weight: 600; margin-top: 10px;
    display: flex; align-items: center;
}}

/* Start/Stop buttons keep their own explicit colors (not theme-driven) */
.st-key-stream_controls_block .stButton button {{
    width: 100%; border-radius: 9px !important; font-weight: 700 !important;
    font-size: 12.5px !important;
}}
.st-key-stream_controls_block div[data-testid="column"]:nth-child(1) .stButton button {{
    background: {ORANGE} !important; color: #150800 !important; border: none !important;
}}
.st-key-stream_controls_block div[data-testid="column"]:nth-child(1) .stButton button:hover {{
    background: {ORANGE_SOFT} !important; color: #150800 !important;
}}
.st-key-stream_controls_block div[data-testid="column"]:nth-child(2) .stButton button {{
    background: transparent !important; color: {SUBTEXT} !important; border: 1px solid {BORDER} !important;
}}

/* ---------- Topbar ---------- */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 18px; margin-bottom: 6px;
}}
.crumb {{ font-size: 13px; color: {SUBTEXT}; display: flex; align-items: center; gap: 8px; }}
.crumb b {{ color: {TEXT}; font-weight: 600; }}
.top-actions {{ display: flex; align-items: center; gap: 10px; }}
.icon-chip {{
    width: 34px; height: 34px; border-radius: 10px; background: {SURFACE_ALT};
    border: 1px solid {BORDER}; display: flex; align-items: center; justify-content: center;
    color: {SUBTEXT};
}}
.avatar-chip {{
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(140deg, {ORANGE}, #C23F00);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 13px;
}}
.live-chip {{
    display: flex; align-items: center; gap: 7px;
    background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 8px 14px; font-size: 12.5px; font-weight: 700; color: {TEXT};
}}

/* ---------- Page header ---------- */
.page-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 2px;
}}
.page-sub {{ color: {SUBTEXT}; font-size: 13.5px; }}

/* ---------- Generic card ---------- */
.card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 20px 22px;
}}
.card-title {{ font-weight: 700; font-size: 15px; color: {TEXT}; display: flex; align-items: center; gap: 8px; }}
.card-caption {{ font-size: 11.5px; color: {SUBTEXT}; margin-top: 1px; }}

/* ---------- KPI cards ---------- */
.kpi {{
    border-radius: 16px; padding: 20px 22px; position: relative;
    border: 1px solid {BORDER}; background: {SURFACE}; min-height: 172px;
    display: flex; flex-direction: column; justify-content: space-between;
}}
.kpi.hero {{
    background: linear-gradient(150deg, #FF5A1F 0%, #E4460D 55%, #B33500 100%);
    border: none; box-shadow: 0 18px 40px -18px rgba(255,90,31,0.55);
}}
.kpi.alertcard {{
    background: linear-gradient(150deg, #3A1220 0%, #2A0D18 100%);
    border: 1px solid rgba(247,106,106,0.3);
}}
.kpi-top {{ display: flex; align-items: center; justify-content: space-between; }}
.kpi-icon {{
    width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; background: {SURFACE_ALT}; color: {ORANGE};
}}
.kpi.hero .kpi-icon {{ background: rgba(255,255,255,0.18); color: #fff; }}
.kpi.alertcard .kpi-icon {{ background: rgba(247,106,106,0.18); color: {RED}; }}
.kpi-label {{ font-weight: 700; font-size: 13px; margin-top: 12px; color: {TEXT}; }}
.kpi.hero .kpi-label {{ color: #fff; }}
.kpi-sub {{ font-size: 11.5px; color: {SUBTEXT}; margin-top: 1px; }}
.kpi.hero .kpi-sub {{ color: rgba(255,255,255,0.75); }}
.kpi-value {{ font-size: 27px; margin-top: 12px; color: {TEXT}; }}
.kpi.hero .kpi-value {{ color: #fff; }}
.kpi-foot {{ display: flex; align-items: center; justify-content: space-between; margin-top: 10px; }}
.trend {{ font-size: 11.5px; font-weight: 700; display: flex; align-items: center; gap: 3px; }}
.trend.pos {{ color: {GREEN}; }}
.trend.neg {{ color: {RED}; }}
.kpi.hero .trend.pos {{ color: #D9FFEE; }}
.kpi-link {{ font-size: 12px; font-weight: 600; color: {SUBTEXT}; }}
.kpi.hero .kpi-link {{ color: rgba(255,255,255,0.85); }}
.kpi-dots {{ color: {SUBTEXT}; }}
.kpi.hero .kpi-dots {{ color: rgba(255,255,255,0.7); }}

/* ---------- Table ---------- */
.feed-wrap {{
    overflow-x: auto; overflow-y: auto; max-height: 480px;
    border-radius: 12px; border: 1px solid {BORDER}; margin-top: 6px;
}}
.atable {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
.atable thead th {{
    position: sticky; top: 0; background: {SURFACE_ALT}; z-index: 1;
    text-align: left; font-size: 11px; letter-spacing: .05em; text-transform: uppercase;
    color: {SUBTEXT}; font-weight: 700; padding: 12px 14px; border-bottom: 1px solid {BORDER};
}}
.atable td {{
    padding: 12px 14px; font-size: 13px; color: {TEXT}; border-bottom: 1px solid {BORDER};
}}
.atable tr:last-child td {{ border-bottom: none; }}
.atable tr:hover td {{ background: {SURFACE_ALT}; }}
.atable th:nth-child(1), .atable td:nth-child(1) {{ width: 22%; }}
.atable th:nth-child(2), .atable td:nth-child(2) {{ width: 26%; }}
.atable th:nth-child(3), .atable td:nth-child(3) {{ width: 26%; }}
.atable th:nth-child(4), .atable td:nth-child(4) {{ width: 26%; }}

.badge {{
    font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; display: inline-flex;
    align-items: center; gap: 5px;
}}
.badge .dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.badge.fraud {{ background: rgba(247,106,106,0.14); color: {RED}; }}
.badge.fraud .dot {{ background: {RED}; }}
.badge.legit {{ background: rgba(62,213,152,0.14); color: {GREEN}; }}
.badge.legit .dot {{ background: {GREEN}; }}

/* ---------- SHAP factor rows ---------- */
.factor-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 4px; border-bottom: 1px solid {BORDER}; font-size: 13px;
}}
.factor-row:last-child {{ border-bottom: none; }}
.factor-name {{ color: {TEXT}; font-weight: 600; }}
.factor-dir {{ color: {SUBTEXT}; font-size: 12px; }}
.factor-val {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; }}
.factor-val.up {{ color: {RED}; }}
.factor-val.down {{ color: {GREEN}; }}

/* ---------- Streamlit widget restyle ---------- */
div[data-baseweb="select"] > div {{
    background: {SURFACE_ALT} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; color: {TEXT} !important; font-size: 13px !important;
}}
[data-testid="stMetricValue"] {{ color: {TEXT}; }}
div[role="radiogroup"] label {{ color: {SUBTEXT} !important; }}
.stSlider label {{ color: {SUBTEXT} !important; font-size: 12.5px !important; }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
hr {{ border-color: {BORDER}; }}

/* ---------- Native bordered containers styled as Orbit "cards" ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-radius: 16px !important;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHED RESOURCE / DATA LOADING  (unchanged logic)
# ============================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model, scaler, and SHAP explainer from disk.
    Wrapped in try/except so a missing or corrupted artifact file shows
    a clean error message instead of a raw traceback to anyone viewing
    the dashboard (important once this is deployed publicly).
    """
    try:
        model = joblib.load("artifacts/champion_model.pkl")
        scaler = joblib.load("artifacts/scaler.pkl")
        explainer = joblib.load("artifacts/shap_explainer.pkl")
        return model, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"Missing artifact file: {e}. Ensure all files are in the artifacts/ folder.")
        st.stop()
    except Exception as e:
        st.error(
            "Could not load the fraud-detection artifacts. "
            f"Check the files in artifacts/ and their compatibility. Details: {e}"
        )
        st.stop()


model, _scaler, explainer = load_artifacts()
# Note: scaler is loaded for completeness (e.g. future manual-entry input)
# but not currently used since the streamed test data is already scaled.


@st.cache_data
def load_stream_data():
    """
    Builds a streaming order that interleaves fraud cases evenly throughout
    the stream, rather than relying on pure random shuffling — which could
    (and did, statistically ~31% of the time) push all fraud cases past
    the first several hundred transactions. This keeps the stream realistic
    (fraud is still sparse) while guaranteeing a reliable demo.
    """
    X_test = joblib.load('artifacts/X_test.pkl')
    y_test = joblib.load('artifacts/y_test.pkl')

    fraud_idx = y_test[y_test == 1].index.tolist()
    legit_idx = y_test[y_test == 0].index.tolist()

    rng = np.random.default_rng(seed=42)
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    interleaved_idx = []
    fraud_pointer = 0

    for i, idx in enumerate(legit_idx):
        interleaved_idx.append(idx)
        if (i + 1) % STREAM_INTERLEAVE_GAP == 0 and fraud_pointer < len(fraud_idx):
            interleaved_idx.append(fraud_idx[fraud_pointer])
            fraud_pointer += 1

    interleaved_idx.extend(fraud_idx[fraud_pointer:])

    return X_test.loc[interleaved_idx], y_test.loc[interleaved_idx]


def badge(val):
    """Renders a colored pill badge for a binary fraud/legit prediction."""
    if val == 1:
        return '<span class="badge fraud"><span class="dot"></span>Fraud</span>'
    return '<span class="badge legit"><span class="dot"></span>Legit</span>'


X_stream, y_stream = load_stream_data()

# ---- Session State ----
if "stream_position" not in st.session_state:
    st.session_state.stream_position = 0
if "transaction_log" not in st.session_state:
    st.session_state.transaction_log = pd.DataFrame(
        columns=["index", "fraud_probability", "prediction", "actual", "amount"]
    )
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "active_view" not in st.session_state:
    st.session_state.active_view = "Dashboard"


def process_batch(batch_size):
    """
    Pulls the next `batch_size` rows from the test set, scores them
    with the champion model, and appends results to the running log.
    """
    start = st.session_state.stream_position
    end = min(start + batch_size, len(X_stream))

    if start >= len(X_stream):
        st.session_state.streaming = False
        return None

    batch_X = X_stream.iloc[start:end]
    batch_y = y_stream.iloc[start:end]

    probs = model.predict_proba(batch_X)[:, 1]
    preds = model.predict(batch_X)

    new_rows = pd.DataFrame(
        {
            "index": batch_X.index,
            "fraud_probability": probs,
            "prediction": preds,
            "actual": batch_y.values,
            "amount": batch_X["scaled_amount"].values,
        }
    )

    st.session_state.transaction_log = pd.concat(
        [st.session_state.transaction_log, new_rows], ignore_index=True
    )
    st.session_state.stream_position = end
    return new_rows


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
        <div class="brand-mark">{icon('shield')}</div>
        <div><div class="brand-name">AegisWall</div><div class="brand-sub">Risk operations</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Real, working navigation (session-state driven, not scroll-links) ----
    # NOTE: uses st.container(key=...) — a genuine DOM wrapper — rather than
    # splitting a raw <div> across two st.markdown() calls, which does NOT
    # actually nest the buttons inside it (each st.markdown() is rendered
    # as its own isolated HTML fragment) and silently breaks CSS scoping.
    NAV_ITEMS = [
        ("Dashboard", "space_dashboard"),
        ("Live Feed", "bolt"),
        ("Model Insights", "track_changes"),
        ("Reports", "description"),
    ]

    with st.container(key="nav_block"):
        for label, mat_icon in NAV_ITEMS:
            is_active = st.session_state.active_view == label
            if st.button(
                label,
                icon=f":material/{mat_icon}:",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_view = label
                st.rerun()

    st.markdown('<div class="nav-section-label">DATA SOURCE</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="navitem"><div class="left">{icon('database')}<span>Held-out test set</span></div>
    <span class="navbadge live">Simulated</span></div>
    """, unsafe_allow_html=True)

    # ---- Functional Stream Controls (logic unchanged) ----
    st.markdown(f"""
    <div class="stream-card">
        <div class="t">{icon('sliders')} Stream Controls</div>
        <div class="d">Simulates a live transaction feed over the held-out test set.</div>
    </div>
    """, unsafe_allow_html=True)

    speed = st.slider("Transactions per batch", 1, 20, 5)
    interval = st.slider("Refresh interval (seconds)", 1, 5, 2)

    with st.container(key="stream_controls_block"):
        col1, col2 = st.columns(2)
        start_btn = col1.button("Start", use_container_width=True)
        stop_btn = col2.button("Stop", use_container_width=True)

    if start_btn:
        st.session_state.streaming = True
    if stop_btn:
        st.session_state.streaming = False

    reset_btn = st.button("Reset Stream", use_container_width=True)
    if reset_btn:
        st.session_state.stream_position = 0
        st.session_state.transaction_log = st.session_state.transaction_log.iloc[0:0]
        st.session_state.streaming = False

    status_cls = "on" if st.session_state.streaming else "off"
    status_txt = "Streaming live" if st.session_state.streaming else "Stream paused"
    st.markdown(f"""
    <div class="status-line"><span class="status-dot {status_cls}"></span>{status_txt}</div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Main dashboard content lives in a fragment so periodic auto-refresh
# (while streaming) only reruns this block, instead of the whole script.
# This avoids the duplicate/ghost-section rendering that a manual
# time.sleep()+st.rerun() loop can cause on fast refresh intervals.
# ----------------------------------------------------------------------------
fragment_interval = interval if st.session_state.streaming else None


@st.fragment(run_every=fragment_interval)
def render_dashboard():
    # ---- Trigger one batch per script run when streaming is active ----
    if st.session_state.streaming:
        process_batch(speed)

    log = st.session_state.transaction_log
    view = st.session_state.active_view

    # ============================================================================
    # TOPBAR
    # ============================================================================
    live_label = "Live" if st.session_state.streaming else "Idle"
    live_dotclass = "on" if st.session_state.streaming else "off"
    st.markdown(f"""
    <div class="topbar">
        <div class="crumb">AegisWall {icon('arrow')} <b>{view}</b></div>
        <div class="top-actions">
            <div class="icon-chip">{icon('search')}</div>
            <div class="icon-chip">{icon('bell')}</div>
            <div class="icon-chip">{icon('refresh')}</div>
            <div class="live-chip"><span class="status-dot {live_dotclass}"></span>{live_label}</div>
            <div class="avatar-chip">S</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================================
    # PAGE HEADER
    # ============================================================================
    PAGE_SUBTITLES = {
        "Dashboard": "Real-time risk scoring over a simulated transaction stream, powered by XGBoost with SHAP-based explainability.",
        "Live Feed": "Every transaction scored so far, most recent first.",
        "Model Insights": "Rolling fraud-rate trend and SHAP explanations for flagged transactions.",
        "Reports": "Download the current session's transaction log and summary report.",
    }
    PAGE_TITLES = {
        "Dashboard": "Live Transaction Monitor",
        "Live Feed": "Live Transaction Feed",
        "Model Insights": "Model Insights",
        "Reports": "Reports",
    }

    st.markdown(f'<div class="page-title">{PAGE_TITLES[view]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{PAGE_SUBTITLES[view]}</div>', unsafe_allow_html=True)
    st.write("")

    # ============================================================================
    # KPI ROW  (values unchanged from original metric logic) — shown on every view
    # ============================================================================
    n_processed = len(log)
    n_flagged = int(log["prediction"].sum()) if len(log) else 0
    fraud_rate = (log["prediction"].mean() * 100) if len(log) else 0.0
    recall_val = (
        (log[log["actual"] == 1]["prediction"].mean() * 100)
        if len(log[log["actual"] == 1])
        else None
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi hero">
            <div class="kpi-top">
                <div class="kpi-icon">{icon('activity')}</div>
                <div class="kpi-dots">{icon('dots')}</div>
            </div>
            <div class="kpi-label">Transactions Processed</div>
            <div class="kpi-sub">Total scored this session</div>
            <div class="kpi-value display-num">{n_processed:,}</div>
            <div class="kpi-foot">
                <div class="trend pos">{icon('up')} {live_label}</div>
                <div class="kpi-link">Live Feed {icon('arrow')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        alert_cls = "kpi alertcard" if n_flagged > 0 else "kpi"
        st.markdown(f"""
        <div class="{alert_cls}">
            <div class="kpi-top">
                <div class="kpi-icon">{icon('alert')}</div>
                <div class="kpi-dots">{icon('dots')}</div>
            </div>
            <div class="kpi-label">Flagged as Fraud</div>
            <div class="kpi-sub">Predicted positive by model</div>
            <div class="kpi-value display-num">{n_flagged:,}</div>
            <div class="kpi-foot">
                <div class="trend neg">{icon('alert')} Risk events</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-top">
                <div class="kpi-icon">{icon('shield-line')}</div>
                <div class="kpi-dots">{icon('dots')}</div>
            </div>
            <div class="kpi-label">Live Fraud Rate</div>
            <div class="kpi-sub">Share of flagged transactions</div>
            <div class="kpi-value display-num">{fraud_rate:.2f}%</div>
            <div class="kpi-foot">
                <div class="kpi-link">Threshold {RISK_THRESHOLD:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        recall_display = f"{recall_val:.1f}%" if recall_val is not None else "N/A"
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-top">
                <div class="kpi-icon">{icon('target')}</div>
                <div class="kpi-dots">{icon('dots')}</div>
            </div>
            <div class="kpi-label">Detection Recall (so far)</div>
            <div class="kpi-sub">Actual fraud cases correctly caught</div>
            <div class="kpi-value display-num">{recall_display}</div>
            <div class="kpi-foot">
                <div class="kpi-link">Champion: XGBoost</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ============================================================================
    # LIVE TRANSACTION FEED  — shown on Dashboard + Live Feed views
    # ============================================================================
    if view in ("Dashboard", "Live Feed"):
        header_html = (
            '<div class="card" style="margin-bottom:0;">'
            '<div style="display:flex; align-items:flex-start; justify-content:space-between;">'
            '<div>'
            f'<div class="card-title">{icon("activity", ORANGE)} Live Transaction Feed</div>'
            '<div class="card-caption">Most recent scored transactions, newest first</div>'
            '</div>'
            f'<span class="navbadge">{min(len(log), 15)} of {len(log)} shown</span>'
            '</div>'
        )

        if len(log) > 0:
            display_log = log.tail(15).iloc[::-1].copy()
            display_log["fraud_probability"] = display_log["fraud_probability"].apply(lambda x: f"{x:.2%}")
            display_log["prediction"] = display_log["prediction"].apply(badge)
            display_log["actual"] = display_log["actual"].apply(badge)

            table_html = display_log[["index", "fraud_probability", "prediction", "actual"]].rename(
                columns={
                    "index": "Txn Index",
                    "fraud_probability": "Fraud Probability",
                    "prediction": "Prediction",
                    "actual": "Actual",
                }
            ).to_html(escape=False, index=False)
            table_html = table_html.replace("<table", "<table class='atable'")

            full_html = header_html + f'<div class="feed-wrap">{table_html}</div>' + '</div>'
            st.markdown(full_html, unsafe_allow_html=True)
        else:
            full_html = header_html + '</div>'
            st.markdown(full_html, unsafe_allow_html=True)
            st.info("Press Start in the sidebar to begin simulating live transactions.")

        st.write("")

    # ============================================================================
    # ROLLING FRAUD RATE + SHAP EXPLANATION  — shown on Dashboard + Model Insights
    # ============================================================================
    if view in ("Dashboard", "Model Insights"):
        c1, c2 = st.columns([1.5, 1.1])

        with c1:
            with st.container(border=True):
                st.markdown(
                    f'<div class="card-title">{icon("target", ORANGE)} Rolling Fraud Detection Rate</div>'
                    '<div class="card-caption">Rolling average of flagged transactions over time</div>',
                    unsafe_allow_html=True,
                )
                st.write("")

                if len(log) >= 5:
                    log_indexed = log.reset_index(drop=True)
                    log_indexed["transaction_number"] = log_indexed.index + 1
                    window = min(20, max(5, len(log_indexed) // 5))
                    log_indexed["rolling_fraud_rate"] = (
                        log_indexed["prediction"].rolling(window=window, min_periods=1).mean() * 100
                    )

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=log_indexed["transaction_number"],
                        y=log_indexed["rolling_fraud_rate"],
                        mode="lines",
                        line=dict(color=ORANGE, width=2.4, shape="spline"),
                        fill="tozeroy",
                        fillcolor="rgba(255,90,31,0.12)",
                        hovertemplate="Txn #%{x}: %{y:.2f}%<extra></extra>",
                    ))
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=6, b=0),
                        height=270,
                        font=dict(color=SUBTEXT, family="Inter", size=11),
                        xaxis=dict(title="Transaction #", showgrid=False, zeroline=False, color=SUBTEXT),
                        yaxis=dict(title="Fraud rate (%)", showgrid=True, gridcolor=BORDER, zeroline=False, color=SUBTEXT),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.caption(f"Window size: {window} transactions (auto-scaled)")
                else:
                    st.caption("Chart will appear once at least 5 transactions have streamed.")

        with c2:
            with st.container(border=True):
                st.markdown(
                    f'<div class="card-title">{icon("eye", ORANGE)} Why Was This Flagged?</div>'
                    '<div class="card-caption">SHAP explanation for the most recently flagged transaction</div>',
                    unsafe_allow_html=True,
                )
                st.write("")

                if len(log) > 0:
                    recent_fraud = log[log["fraud_probability"] >= RISK_THRESHOLD]

                    if len(recent_fraud) > 0:
                        latest_flagged = recent_fraud.iloc[-1]
                        txn_idx = latest_flagged["index"]

                        st.warning(
                            f"Transaction `{txn_idx}` flagged with "
                            f"{latest_flagged['fraud_probability']:.2%} fraud probability."
                        )

                        txn_data = X_stream.loc[[txn_idx]]
                        shap_vals = explainer.shap_values(txn_data)

                        # SHAP's return shape differs across SHAP/XGBoost versions.
                        # Normalize it to a single 1-D vector for this binary classifier.
                        if isinstance(shap_vals, list):
                            shap_row = np.asarray(shap_vals[-1])[0]
                        else:
                            shap_arr = np.asarray(shap_vals)
                            if shap_arr.ndim == 3:
                                # Common multiclass/binary-output shape:
                                # (samples, features, outputs).
                                shap_row = shap_arr[0, :, -1]
                            elif shap_arr.ndim == 2:
                                shap_row = shap_arr[0]
                            else:
                                shap_row = shap_arr.reshape(-1)

                        if len(shap_row) != len(txn_data.columns):
                            raise ValueError(
                                "SHAP output does not match the transaction feature count. "
                                f"Expected {len(txn_data.columns)}, got {len(shap_row)}."
                            )

                        contrib = pd.DataFrame(
                            {"feature": txn_data.columns, "shap_value": shap_row}
                        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).head(5)

                        rows_html = ""
                        for _, row in contrib.iterrows():
                            up = row["shap_value"] > 0
                            direction = "increased" if up else "decreased"
                            cls = "up" if up else "down"
                            rows_html += f"""
                            <div class="factor-row">
                                <div><span class="factor-name">{row['feature']}</span>
                                    <div class="factor-dir">{direction} fraud risk</div></div>
                                <div class="factor-val {cls}">{row['shap_value']:+.3f}</div>
                            </div>"""
                        st.markdown(f'<div style="margin-top:4px;">{rows_html}</div>', unsafe_allow_html=True)

                        with st.expander("View SHAP waterfall plot"):
                            expected_value = np.asarray(explainer.expected_value)
                            if expected_value.ndim == 0:
                                base_value = float(expected_value)
                            else:
                                base_value = float(expected_value.reshape(-1)[-1])

                            explanation = shap.Explanation(
                                values=shap_row,
                                base_values=base_value,
                                data=txn_data.iloc[0].values,
                                feature_names=txn_data.columns.tolist(),
                            )
                            plt.style.use("dark_background")
                            fig = plt.figure(figsize=(8, 4.5))
                            fig.patch.set_facecolor(SURFACE)
                            shap.plots.waterfall(explanation, show=False)
                            st.pyplot(fig, bbox_inches="tight")
                            plt.close(fig)
                            plt.style.use("default")
                    else:
                        st.caption("No high-risk transactions detected yet in this stream.")
                else:
                    st.caption("Start the stream to see live risk explanations.")

    # ============================================================================
    # REPORTS  — real, downloadable session exports
    # ============================================================================
    if view == "Reports":
        with st.container(border=True):
            st.markdown(
                f'<div class="card-title">{icon("file", ORANGE)} Session Reports</div>'
                '<div class="card-caption">Export the current streaming session for offline review</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            if len(log) == 0:
                st.info("Start the stream in the sidebar to generate a report.")
            else:
                generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                summary_lines = [
                    "# AegisWall — Session Report",
                    f"Generated: {generated_at}",
                    "",
                    "## Summary",
                    f"- Transactions processed: {n_processed:,}",
                    f"- Flagged as fraud: {n_flagged:,}",
                    f"- Live fraud rate: {fraud_rate:.2f}%",
                    f"- Detection recall (so far): {recall_display}",
                    f"- Risk threshold: {RISK_THRESHOLD:.0%}",
                    "- Model: XGBoost (champion), PR-AUC 0.828 / ROC-AUC 0.974",
                    "- Explainability: SHAP TreeExplainer",
                    "",
                    "## Notes",
                    "This is a simulation over a held-out test set, not a live production feed.",
                    "Fraud cases are interleaved evenly through the stream for demo reliability "
                    "(one roughly every 150 legitimate transactions), rather than shuffled purely "
                    "at random.",
                ]
                summary_md = "\n".join(summary_lines)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "Download transaction log (CSV)",
                        data=log.to_csv(index=False),
                        file_name=f"fraud_shield_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl2:
                    st.download_button(
                        "Download summary report (Markdown)",
                        data=summary_md,
                        file_name=f"fraud_shield_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                st.write("")
                st.caption(f"Preview — {len(log):,} transactions in this session")
                st.dataframe(log, use_container_width=True, height=320)


render_dashboard()