"""
Streamlit data-storytelling dashboard for the Dynamic Pricing project.

Single scrolling page styled like a walk through a grocery store. The
sidebar lists "aisles" — clicking one jumps to that anchor on the same
page. No multi-page mode; the narrative is meant to read top-to-bottom.

Run locally:
    cd code/
    streamlit run dashboard.py

For Streamlit Community Cloud, point the app at code/dashboard.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fresh Pricing in Missouri",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Grocery palette
PRODUCE  = "#2D5016"   # deep leafy green
LEAF     = "#5B8C3F"   # softer green
BAG      = "#8B4513"   # paper bag brown
APPLE    = "#C44E52"   # apple red
PEACH    = "#E8956F"   # warm peach
CREAM    = "#FAF7F2"   # cream paper
PAPER    = "#F5F0E8"   # parchment
INK      = "#2C2C2C"   # ink

STRAT_PALETTE = {"No Discount": "#5B8AA6", "Fixed 20%": "#E07A5F", "Dynamic": LEAF}
APP_FONT = "'Source Sans Pro', system-ui, sans-serif"
NUMBER_FONT = "'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif"

pio.templates["fresh_source_sans"] = go.layout.Template(pio.templates["simple_white"])
pio.templates["fresh_source_sans"].layout.font = dict(family=APP_FONT, color=INK)
pio.templates["fresh_source_sans"].layout.title = dict(font=dict(family=APP_FONT, color=PRODUCE))
PLOTLY_TEMPLATE = "fresh_source_sans"

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=block');
    @font-face {{
        font-family: 'Material Symbols Rounded';
        font-style: normal;
        font-weight: 400;
        font-display: block;
        src: url('/static/media/MaterialSymbols-Rounded.BK8hQpFn.woff2') format('woff2');
    }}

    /* ---------- Global ---------- */
    html, body, .stApp {{
        font-family: 'Source Sans Pro', system-ui, -apple-system, 'Helvetica Neue', sans-serif;
        color: {INK};
        background-color: {CREAM} !important;
    }}
    .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {{
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        color: {INK};
    }}
    .stApp div:not(.material-icons):not(.material-icons-outlined):not(.material-symbols-outlined):not(.material-symbols-rounded):not(.material-symbols-sharp):not([class*="material-icons"]):not([class*="material-symbols"]):not([data-testid="stIconMaterial"]),
    .stApp span:not(.material-icons):not(.material-icons-outlined):not(.material-symbols-outlined):not(.material-symbols-rounded):not(.material-symbols-sharp):not([class*="material-icons"]):not([class*="material-symbols"]):not([data-testid="stIconMaterial"]),
    .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp textarea, .stApp select, .stApp table, .stApp th, .stApp td,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stExpander"],
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"] {{
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
    }}
    .stApp strong, .stApp b, .stApp em, .stApp small, .stApp caption {{
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
    }}
    .material-icons, .material-icons-outlined, .material-symbols-outlined,
    .material-symbols-rounded, .material-symbols-sharp,
    [class*="material-icons"], [class*="MaterialIcons"], [class*="material-symbols"],
    [data-testid="stIconMaterial"],
    .stApp [data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' !important;
    }}
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {{
        font-size: 0 !important;
        width: 1.65rem !important;
        height: 1.65rem !important;
        line-height: 1 !important;
    }}
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::before,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::before {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.65rem;
        height: 1.65rem;
        color: {BAG};
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1;
    }}
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::before {{
        content: "\\00BB";
    }}
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::before {{
        content: "\\00AB";
    }}

    /* ---------- Headings ---------- */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
        color: {PRODUCE} !important;
        letter-spacing: 0;
        font-weight: 800 !important;
    }}
    h1 {{ font-size: 3.35rem !important; line-height: 1.02 !important; margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }}
    h2 {{ font-size: 1.85rem !important; line-height: 1.2 !important; padding-top: 0.5rem !important; }}
    h3 {{ font-size: 1.35rem !important; line-height: 1.3 !important; }}
    h4 {{ font-size: 1.1rem !important; }}

    /* ---------- Section divider — clean ruled line ---------- */
    .section-divider {{
        border: none;
        border-top: 1px solid #E0D9CE;
        margin: 2.8rem 0 1.8rem 0;
    }}

    /* ---------- Hero ---------- */
    .hero {{
        background: linear-gradient(135deg, {PAPER} 0%, {CREAM} 100%);
        border: 1px solid rgba(224, 217, 206, 0.9);
        padding: 2.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 26px rgba(45, 80, 22, 0.06);
    }}
    .hero-title {{
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: {PRODUCE};
        margin: 0 0 0.5rem 0;
        line-height: 1.1;
        letter-spacing: 0;
    }}
    .hero-sub {{
        color: {BAG};
        font-size: 1.05rem;
        font-style: italic;
        margin: 0 0 1rem 0;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
    }}
    .hero-stat {{
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif;
        font-size: 4.4rem;
        font-weight: 800;
        color: {APPLE};
        line-height: 1;
        margin: 0.5rem 0 0.4rem 0;
        letter-spacing: 0;
    }}
    .hero-stat-cap {{
        color: {INK};
        font-size: 0.95rem;
        font-weight: 500;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
    }}
    .hero-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.15rem 0 1.45rem 0;
    }}
    .hero-kpi-card {{
        min-height: 124px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(231, 222, 209, 0.85);
        border-radius: 18px;
        padding: 1.15rem 1.2rem 1.05rem 1.2rem;
        box-shadow: 0 12px 28px rgba(45, 80, 22, 0.07);
        overflow: hidden;
    }}
    .hero-kpi-label {{
        color: {BAG};
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        line-height: 1.2;
    }}
    .hero-kpi-value {{
        color: {PRODUCE};
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1.05;
    }}
    .hero-kpi-delta {{
        width: fit-content;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.2;
    }}
    .hero-kpi-good {{
        color: {PRODUCE};
        background: #EEF6EA;
    }}
    .hero-kpi-warn {{
        color: #913B42;
        background: #FBE8E7;
    }}
    @media (max-width: 800px) {{
        .hero-kpi-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    /* ---------- Aisle label (small uppercase pill) ---------- */
    .aisle-marker {{
        display: inline-block;
        background: {PRODUCE};
        color: {CREAM};
        padding: 0.28rem 0.85rem;
        border-radius: 14px;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        text-transform: uppercase;
    }}

    /* ---------- Callouts ---------- */
    .callout {{
        border: 1px solid rgba(232, 149, 111, 0.28);
        background: #FFF8F0;
        padding: 1rem 1.25rem;
        border-radius: 14px;
        margin: 1.25rem 0;
        color: {INK};
        font-size: 0.96rem;
        line-height: 1.55;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        box-shadow: 0 10px 24px rgba(139, 69, 19, 0.06);
    }}
    .callout-good {{
        border-color: rgba(91, 140, 63, 0.24);
        background: #F4F8EE;
    }}
    .callout-warn {{
        border-color: rgba(232, 149, 111, 0.28);
        background: #FFF3EB;
    }}
    .callout b {{ color: {PRODUCE}; }}

    .side-note {{
        margin: 5.3rem 0 0 0.15rem;
        padding: 0.1rem 0 0.1rem 0.9rem;
        border-left: 2px solid rgba(91, 140, 63, 0.36);
        color: {BAG};
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        font-size: 0.92rem;
        line-height: 1.55;
    }}
    .side-note-title {{
        display: block;
        margin-bottom: 0.45rem;
        color: {PRODUCE};
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .side-note em {{
        color: {INK};
    }}

    /* ---------- Process stages ---------- */
    .stage-flow {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 3.2rem minmax(0, 1fr);
        grid-template-rows: auto 2.45rem auto 2.45rem auto;
        gap: 0.2rem 0.75rem;
        align-items: center;
        margin: 1.25rem 0 1.35rem 0;
    }}
    .stage-card {{
        display: grid;
        grid-template-columns: 3.35rem minmax(0, 1fr);
        gap: 1rem;
        align-items: center;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(231, 222, 209, 0.88);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 28px rgba(45, 80, 22, 0.06);
    }}
    .stage-card-1 {{ grid-column: 1; grid-row: 1; }}
    .stage-card-2 {{ grid-column: 3; grid-row: 1; }}
    .stage-card-3 {{ grid-column: 3; grid-row: 3; }}
    .stage-card-4 {{ grid-column: 1; grid-row: 3; }}
    .stage-card-5 {{ grid-column: 1; grid-row: 5; }}
    .stage-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.75rem;
        height: 2.75rem;
        border-radius: 999px;
        background: {PRODUCE};
        color: {CREAM};
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(45, 80, 22, 0.14);
    }}
    .stage-title {{
        margin: 0 0 0.38rem 0;
        color: {PRODUCE};
        font-size: 1.14rem;
        font-weight: 800;
        line-height: 1.2;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
    }}
    .stage-body {{
        margin: 0;
        color: {INK};
        font-size: 0.9rem;
        line-height: 1.46;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
    }}
    .stage-body b {{
        color: {PRODUCE};
        font-weight: 800;
    }}
    .stage-arrow {{
        position: relative;
        width: 100%;
        height: 2.45rem;
        font-size: 0;
    }}
    .stage-arrow-right {{ grid-column: 2; grid-row: 1; }}
    .stage-arrow-down-1 {{ grid-column: 3; grid-row: 2; }}
    .stage-arrow-left {{ grid-column: 2; grid-row: 3; }}
    .stage-arrow-down-2 {{ grid-column: 1; grid-row: 4; }}
    .stage-arrow-horizontal::before {{
        content: '';
        position: absolute;
        left: 0;
        right: 0;
        top: 50%;
        height: 3px;
        border-radius: 999px;
    }}
    .stage-arrow-right::before {{
        background: linear-gradient(to right, rgba(91, 140, 63, 0.16), rgba(91, 140, 63, 0.62));
    }}
    .stage-arrow-left::before {{
        background: linear-gradient(to left, rgba(91, 140, 63, 0.16), rgba(91, 140, 63, 0.62));
    }}
    .stage-arrow-horizontal::after {{
        content: '';
        position: absolute;
        top: 50%;
        width: 0.68rem;
        height: 0.68rem;
    }}
    .stage-arrow-right::after {{
        right: 0;
        border-top: 3px solid rgba(91, 140, 63, 0.68);
        border-right: 3px solid rgba(91, 140, 63, 0.68);
        transform: translateY(-50%) rotate(45deg);
    }}
    .stage-arrow-left::after {{
        left: 0;
        border-left: 3px solid rgba(91, 140, 63, 0.68);
        border-bottom: 3px solid rgba(91, 140, 63, 0.68);
        transform: translateY(-50%) rotate(45deg);
    }}
    .stage-arrow-down::before {{
        content: '';
        position: absolute;
        left: 50%;
        top: 0.18rem;
        width: 3px;
        height: 1.75rem;
        border-radius: 999px;
        background: linear-gradient(to bottom, rgba(91, 140, 63, 0.16), rgba(91, 140, 63, 0.62));
        transform: translateX(-50%);
    }}
    .stage-arrow-down::after {{
        content: '';
        position: absolute;
        left: 50%;
        top: 1.38rem;
        width: 0.68rem;
        height: 0.68rem;
        border-right: 3px solid rgba(91, 140, 63, 0.68);
        border-bottom: 3px solid rgba(91, 140, 63, 0.68);
        transform: translateX(-50%) rotate(45deg);
    }}
    @media (max-width: 900px) {{
        .stage-flow {{
            grid-template-columns: 1fr;
            grid-template-rows: none;
            gap: 0.25rem;
        }}
        .stage-card {{
            grid-template-columns: 1fr;
            gap: 0.7rem;
        }}
        .stage-card-1, .stage-card-2, .stage-card-3, .stage-card-4, .stage-card-5,
        .stage-arrow-right, .stage-arrow-down-1, .stage-arrow-left, .stage-arrow-down-2 {{
            grid-column: 1;
            grid-row: auto;
        }}
        .stage-arrow-horizontal::before {{
            left: 50%;
            right: auto;
            top: 0.18rem;
            width: 2px;
            height: 1.75rem;
            background: linear-gradient(to bottom, rgba(91, 140, 63, 0.16), rgba(91, 140, 63, 0.62));
            transform: translateX(-50%);
        }}
        .stage-arrow-right::after, .stage-arrow-left::after {{
            left: 50%;
            right: auto;
            top: 1.38rem;
            border: none;
            border-right: 2px solid rgba(91, 140, 63, 0.62);
            border-bottom: 2px solid rgba(91, 140, 63, 0.62);
            transform: translateX(-50%) rotate(45deg);
        }}
    }}

    /* ═══════════════════════════════════════════════
       SIDEBAR — tree view + journey line
       ═══════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {{
        background: #F3EEE6 !important;
        border-right: none;
    }}
    section[data-testid="stSidebar"] > div {{
        background: #F4EDE3 !important;
        border: 1px solid #E2D5C5;
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(45, 80, 22, 0.06);
        margin: 0.2rem 0.45rem 0.55rem 0.55rem;
        padding-top: 0 !important;
        max-height: calc(100vh - 0.75rem);
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-width: thin;
        scrollbar-color: rgba(91, 140, 63, 0.34) transparent;
    }}
    section[data-testid="stSidebar"] > div::-webkit-scrollbar {{
        width: 8px;
    }}
    section[data-testid="stSidebar"] > div::-webkit-scrollbar-track {{
        background: transparent;
    }}
    section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {{
        background: rgba(91, 140, 63, 0.32);
        border-radius: 999px;
        border: 2px solid #F4EDE3;
    }}
    section[data-testid="stSidebar"] .element-container {{
        margin-bottom: 0 !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        padding: 0 !important;
    }}

    /* Logo block */
    .sb-logo {{
        padding: 0.65rem 0.95rem 0.72rem 1.05rem;
        border-bottom: 1px solid rgba(139, 69, 19, 0.12);
        margin-bottom: 0.24rem;
        background: transparent;
        border-radius: 0;
    }}
    .sb-logo-mark {{
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        color: white;
        background: {PRODUCE};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        margin-bottom: 0.34rem;
        letter-spacing: 0.03em;
        box-shadow: 0 8px 18px rgba(45, 80, 22, 0.16);
    }}
    .sb-logo-name {{
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        font-size: 1.04rem;
        font-weight: 800;
        color: {PRODUCE};
        line-height: 1.1;
        display: block;
    }}
    .sb-logo-sub {{
        font-size: 0.63rem;
        color: {BAG};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 0.18rem;
        display: block;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
    }}

    /* Journey line — fixed to left edge of sidebar */
    .journey-track-outer {{
        position: fixed;
        left: 18px;
        top: 150px;
        bottom: 58px;
        width: 2px;
        background: rgba(139, 69, 19, 0.14);
        border-radius: 999px;
        z-index: 999;
        pointer-events: none;
    }}
    .journey-fill {{
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 0%;
        background: linear-gradient(to bottom, {PRODUCE}, {LEAF});
        border-radius: 999px;
        transition: height 0.25s ease-out;
    }}
    .tree-group {{
        margin-bottom: 0.08rem;
        padding: 0 0.5rem;
    }}
    .tree-group-label {{
        font-size: 0.61rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {PRODUCE};
        padding: 0.52rem 0.48rem 0.2rem 0.48rem;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        display: block;
    }}
    .tree-children {{
        position: relative;
        padding-left: 0.88rem;
        margin-bottom: 0.24rem;
    }}
    .tree-children::before {{
        content: '';
        position: absolute;
        left: 0.2rem;
        top: 0.52rem;
        bottom: 0.58rem;
        width: 2px;
        background: rgba(91, 140, 63, 0.18);
        border-radius: 999px;
    }}
    .tree-link {{
        display: block;
        padding: 0.34rem 0.56rem 0.34rem 0.72rem;
        margin: 0.05rem 0;
        color: #5F431F;
        text-decoration: none !important;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 10px;
        position: relative;
        transition: background 0.12s ease, color 0.12s ease, padding-left 0.12s ease,
                    box-shadow 0.12s ease;
        font-family: 'Source Sans Pro', system-ui, sans-serif;
        line-height: 1.35;
    }}
    .tree-link::before {{
        content: '';
        position: absolute;
        left: -0.5rem;
        top: 50%;
        transform: translateY(-50%);
        width: 0.32rem;
        height: 2px;
        border-radius: 999px;
        background: rgba(139, 69, 19, 0.22);
    }}
    .tree-link:hover, .tree-link:focus {{
        background: rgba(255, 253, 252, 0.68) !important;
        color: {PRODUCE} !important;
        box-shadow: 0 6px 16px rgba(139, 69, 19, 0.08);
        padding-left: 0.9rem;
        text-decoration: none !important;
        outline: none;
    }}
    .tree-link-active,
    .tree-link[aria-current="true"] {{
        background: rgba(255, 253, 252, 0.9) !important;
        color: {PRODUCE} !important;
        box-shadow: 0 7px 18px rgba(45, 80, 22, 0.1);
        font-weight: 800;
        padding-left: 0.84rem;
    }}
    .tree-link-active::before,
    .tree-link[aria-current="true"]::before {{
        background: {LEAF};
        width: 0.48rem;
    }}
    .tree-link:visited {{
        color: #5F431F;
    }}

    /* ---------- Layout ---------- */
    .block-container {{
        padding-top: 1.5rem !important;
        max-width: 1180px !important;
    }}

    /* ---------- Streamlit elements ---------- */
    .stDataFrame, .stTable {{
        font-size: 0.9rem;
    }}
    [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #E7DED1;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 122px;
        box-shadow: 0 8px 22px rgba(45, 80, 22, 0.08);
    }}
    .stMetric label {{
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
        font-size: 0.78rem !important;
        color: {BAG} !important;
        letter-spacing: 0.04em;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif !important;
        color: {PRODUCE} !important;
        font-weight: 800 !important;
        letter-spacing: 0 !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'Avenir Next', Avenir, 'Source Sans Pro', system-ui, sans-serif !important;
        font-weight: 700 !important;
    }}
    .stButton > button {{
        font-family: 'Source Sans Pro', system-ui, sans-serif !important;
        font-weight: 600;
        border-radius: 4px;
    }}
    .stMarkdown p, .stMarkdown li {{
        line-height: 1.65;
        color: {INK};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {BAG} !important;
        font-style: italic;
    }}

    /* Plotly charts — match background */
    .js-plotly-plot, .plotly {{
        background-color: transparent !important;
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Journey-line scroll tracker — injected once per render into the main page
# ─────────────────────────────────────────────────────────────────────────────
JOURNEY_JS = """
<script>
(function(){
    var fill = null;
    var navLinks = null;

    function getFill(){
        if(!fill) fill = document.getElementById('journey-line-fill');
        return fill;
    }

    function getNavLinks(){
        if(!navLinks || !navLinks.length){
            navLinks = Array.prototype.slice.call(document.querySelectorAll('a.tree-link[href^="#"]'));
        }
        return navLinks;
    }

    function getAnchorForLink(link){
        var hash = link.getAttribute('href') || '';
        if(hash.charAt(0) !== '#') return null;
        var name = decodeURIComponent(hash.slice(1));
        return document.getElementsByName(name)[0] || document.getElementById(name);
    }

    function updateActiveLink(){
        var links = getNavLinks();
        if(!links.length) return;

        var threshold = Math.min(window.innerHeight * 0.32, 220);
        var active = links[0];
        var bestTop = -Infinity;

        links.forEach(function(link){
            var anchor = getAnchorForLink(link);
            if(!anchor) return;
            var top = anchor.getBoundingClientRect().top;
            if(top <= threshold && top > bestTop){
                bestTop = top;
                active = link;
            }
        });

        links.forEach(function(link){
            var isActive = link === active;
            link.classList.toggle('tree-link-active', isActive);
            if(isActive){
                link.setAttribute('aria-current', 'true');
            } else {
                link.removeAttribute('aria-current');
            }
        });
    }

    function update(){
        var f = getFill();
        if(!f) return;

        // Streamlit's scroll container varies across versions — try each
        var top = 0, height = 0;
        var candidates = [
            document.querySelector('section.main'),
            document.querySelector('[data-testid="stMain"]'),
            document.querySelector('[data-testid="stAppViewContainer"] > section'),
            document.querySelector('.main'),
        ];
        for(var i = 0; i < candidates.length; i++){
            var c = candidates[i];
            if(!c) continue;
            var h = c.scrollHeight - c.clientHeight;
            if(h > 50){
                top    = c.scrollTop;
                height = h;
                if(top > 0) break;
            }
        }
        if(height === 0){
            top    = window.pageYOffset || document.documentElement.scrollTop || 0;
            height = document.documentElement.scrollHeight - window.innerHeight;
        }

        var pct = height > 0 ? Math.min(100, Math.max(0, top / height * 100)) : 0;
        f.style.height = pct + '%';
        updateActiveLink();
    }

    window.addEventListener('scroll', update, {passive: true});
    document.addEventListener('scroll', update, {passive: true, capture: true});
    window.addEventListener('hashchange', updateActiveLink);

    function attachScrollListeners(){
        document.querySelectorAll('section, div, main').forEach(function(el){
            try {
                var s = window.getComputedStyle(el);
                if((s.overflowY === 'auto' || s.overflowY === 'scroll') &&
                   el.scrollHeight > el.clientHeight + 100){
                    el.addEventListener('scroll', update, {passive: true});
                }
            } catch(e){}
        });
        getNavLinks().forEach(function(link){
            if(link.dataset.activeBound) return;
            link.dataset.activeBound = 'true';
            link.addEventListener('click', function(){
                setTimeout(updateActiveLink, 120);
            });
        });
    }

    setTimeout(attachScrollListeners, 1200);
    setTimeout(attachScrollListeners, 3000);
    setTimeout(update, 400);
    setTimeout(updateActiveLink, 600);
    setInterval(update, 700);   // polling fallback
})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = config.OUTPUTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_manifest() -> dict | None:
    path = config.OUTPUTS_DIR / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_spoilage_model():
    path = config.MODELS_DIR / "spoilage_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def anchor(name: str) -> None:
    """Drop an HTML anchor so the sidebar can jump to it."""
    st.markdown(f"<a name='{name}'></a>", unsafe_allow_html=True)


def divider() -> None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


def aisle_label(text: str) -> None:
    st.markdown(f"<div class='aisle-marker'>{text}</div>", unsafe_allow_html=True)


def callout(text: str, variant: str = "neutral") -> None:
    klass = {"good": "callout callout-good",
             "warn": "callout callout-warn",
             "honest": "callout"}.get(variant, "callout")
    st.markdown(f"<div class='{klass}'>{text}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — tree view with journey-line progress
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR_GROUPS = [
    ("Walking In", [
        ("welcome", "Welcome"),
        ("problem", "The Problem"),
    ]),
    ("How It Works", [
        ("approach", "Our Approach"),
        ("datasets", "The Data"),
    ]),
    ("The Results", [
        ("showdown", "Strategy Showdown"),
        ("aisles",   "Aisle by Aisle"),
        ("expiry",   "Sell-by-date Logic"),
        ("register", "Discount Distribution"),
    ]),
    ("Stress & Sensitivity", [
        ("stress",      "Near-expiry Stress"),
        ("sensitivity", "Elasticity Sensitivity"),
    ]),
    ("Interactive", [
        ("whatif",  "Try It Yourself"),
        ("models",  "Under the Hood"),
    ]),
    ("Closing", [
        ("limits",   "Honest Limits"),
        ("checkout", "Final Receipt"),
    ]),
]

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            "<div class='sb-logo'>"
            "<div class='sb-logo-mark'>FP</div>"
            "<span class='sb-logo-name'>Fresh Pricing</span>"
            "<span class='sb-logo-sub'>Missouri · FSE 570</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Journey line — fixed-position, updated by JS
        st.markdown(
            "<div class='journey-track-outer'>"
            "<div id='journey-line-fill' class='journey-fill'></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Tree nav — one call per group
        for group_title, items in SIDEBAR_GROUPS:
            links = "".join(
                f"<a class='tree-link' href='#{slug}'>{label}</a>"
                for slug, label in items
            )
            st.markdown(
                f"<div class='tree-group'>"
                f"<span class='tree-group-label'>{group_title}</span>"
                f"<div class='tree-children'>{links}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_modern_sidebar():
    return
    """
            "<span class='fresh-brand-sub'>Missouri · FSE 570</span></div>"
            "</div>"
            f"<div class='fresh-nav'>{''.join(nav_blocks)}</div>"
            "<div class='fresh-sidebar-note'>Click a section to open its aisle list. Only one drawer stays open at a time.</div>"
            "</div></div>"
            f"{SIDEBAR_SCRIPT}",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
    """
# Sections (rendered top-to-bottom on a single long page)
# ─────────────────────────────────────────────────────────────────────────────
def section_welcome():
    anchor("welcome")
    delta = load_csv("strategy_delta_table.csv")
    manifest = load_manifest()

    st.markdown("""
<h1>Fresh Pricing</h1>
<p class='hero-sub'>A data-driven grocery dashboard that knows when to mark something down, and when to leave it alone.</p>
""", unsafe_allow_html=True)

    if delta is not None:
        try:
            dyn = float(delta.set_index("Strategy").loc["Dynamic", "Profit_USD"])
            base = float(delta.set_index("Strategy").loc["No Discount", "Profit_USD"])
            multiple = dyn / base if base else None
            waste_pct = float(delta.set_index("Strategy").loc["Dynamic", "Waste_Pct"])
            sellt = float(delta.set_index("Strategy").loc["Dynamic", "Sell_Through_Pct"])
        except Exception:
            multiple = None
            waste_pct = sellt = None

        if multiple is not None:
            waste_delta = waste_pct - 33.5
            sell_delta = sellt - 66
            waste_delta_class = "hero-kpi-good" if waste_delta <= 0 else "hero-kpi-warn"
            sell_delta_class = "hero-kpi-good" if sell_delta >= 0 else "hero-kpi-warn"
            st.markdown(f"""
  <div class='hero-stat'>{multiple:.1f}x</div>
  <div class='hero-stat-cap'>more profit than a no-discount baseline · 60-day out-of-sample test</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class='hero-kpi-grid'>
  <div class='hero-kpi-card'>
    <div class='hero-kpi-label'>Profit (Dynamic)</div>
    <div class='hero-kpi-value'>${dyn:,.0f}</div>
    <div class='hero-kpi-delta hero-kpi-good'>+${dyn - base:,.0f} vs baseline</div>
  </div>
  <div class='hero-kpi-card'>
    <div class='hero-kpi-label'>Waste rate</div>
    <div class='hero-kpi-value'>{waste_pct:.1f}%</div>
    <div class='hero-kpi-delta {waste_delta_class}'>{waste_delta:+.1f} pp vs baseline</div>
  </div>
  <div class='hero-kpi-card'>
    <div class='hero-kpi-label'>Sell-through</div>
    <div class='hero-kpi-value'>{sellt:.0f}%</div>
    <div class='hero-kpi-delta {sell_delta_class}'>{sell_delta:+.0f} pp vs baseline</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
This is what a grocery chain could actually deploy. For every SKU on every
day, the system gives one decision: how much, if anything, to mark this item
down. The decision pulls together a demand forecast, the customer's price
sensitivity, and the chance the item spoils before it sells. **Walk through
the aisles below** to see how it works, why it works, and where it doesn't.
""")


def section_problem():
    divider()
    anchor("problem")
    aisle_label("Aisle 1 · The Problem")
    st.markdown("## 10 to 15% of every grocery store walks straight into the dumpster")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
That's the industry baseline for perishable food spoilage, and it happens
because markdown decisions are usually **manual, ad-hoc, and reactive**.

Three failure modes show up over and over:

- **Too late.** The item spoils before anyone marks it down.
- **Too early.** Margin gets sacrificed on items that would have sold at full price.
- **Uniform.** A flat 20% off the whole shelf, regardless of expiry or demand.

The first two waste food. The third one is the interesting one: it actually
**destroys profit** versus doing nothing at all, and we'll show that in a
later aisle.

The right answer isn't more aggressive discounting. It's **targeted**
discounting. Deep cuts where spoilage risk is high *and* customers actually
respond to a lower price. No cuts at all where the shelf life is plenty and
people would have bought it anyway.
""")
    with c2:
        st.markdown(f"""
<aside class='side-note'>
<span class='side-note-title'>Why this matters beyond the spreadsheet</span>
Food waste is the third-largest source of greenhouse gas emissions globally
(<i>Project Drawdown, 2020</i>). Reducing grocery shrinkage isn't only a
margin story. It's also a sustainability story.
</aside>
""", unsafe_allow_html=True)


def section_approach():
    divider()
    anchor("approach")
    aisle_label("Aisle 2 · How we built it")
    st.markdown("## Five stages, from raw transactions to a daily markdown recommendation")

    st.markdown("""
The system is a chain of five stages. Each stage feeds the next. The whole
thing sits inside a strict train, validation, and holdout split, so when we
say a number is out-of-sample, it actually is.
""")

    stages = [
        ("1", "Demand forecasting",
         "LightGBM beat Prophet and Holt-Winters on a 30-day validation window. "
         "We picked the winner by **directional accuracy** (62% versus roughly "
         "50% coin-flip) because the optimizer needs to know whether demand is "
         "going up or down, not just hit the right average."),
        ("2", "Price elasticity",
         "Log-log regression on Dunnhumby with **store + week fixed effects**. "
         "Every department turned out to be price-inelastic, but the spread "
         "matters: Produce ( -0.23 ) is about four times more responsive than "
         "Meat ( -0.06 ). That alone tells us a flat markdown can't be right."),
        ("3", "Spoilage classifier",
         "Four classifiers compared: Logistic Regression, Random Forest, "
         "XGBoost, and LightGBM. LightGBM wins (AUC 0.85, Brier 0.15). The "
         "predicted probability is calibrated against the empirical waste "
         "fraction so it translates into expected wasted units, not just a yes-or-no."),
        ("4", "Optimizer",
         "For every SKU on every day, we score five candidate discounts "
         "(0, 10, 20, 30, 40 percent) and pick the one that **maximises "
         "expected profit** above a 5% margin floor. Items within five days of "
         "expiry use a stronger -2.0 elasticity to model bargain-hunter "
         "behaviour."),
        ("5", "60-day out-of-sample simulation",
         "Three strategies are compared on the held-out final 60 days: No "
         "Discount, Fixed 20%, and Dynamic. None of the models have ever seen "
         "this data."),
    ]
    def bold_markdown(text: str) -> str:
        parts = text.split("**")
        return "".join(f"<b>{part}</b>" if i % 2 else part for i, part in enumerate(parts))

    stage_blocks = []
    arrow_classes = {
        0: "stage-arrow stage-arrow-horizontal stage-arrow-right",
        1: "stage-arrow stage-arrow-down stage-arrow-down-1",
        2: "stage-arrow stage-arrow-horizontal stage-arrow-left",
        3: "stage-arrow stage-arrow-down stage-arrow-down-2",
    }
    for index, (num, title, body) in enumerate(stages):
        stage_blocks.append(
            f"<div class='stage-card stage-card-{num}'>"
            f"<div class='stage-num'>{num}</div>"
            f"<div><div class='stage-title'>{title}</div>"
            f"<p class='stage-body'>{bold_markdown(body)}</p></div>"
            f"</div>"
        )
        if index < len(stages) - 1:
            stage_blocks.append(f"<div class='{arrow_classes[index]}'></div>")

    st.markdown(
        f"<div class='stage-flow'>{''.join(stage_blocks)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""
Every model is trained only on data before the holdout window. The final 60
days stay sealed off until stage 5. So the $1.99M profit number isn't a
retrospective replay of data the models already learned from. It's what they
would have produced on data they had never seen.
""")


def section_datasets():
    divider()
    anchor("datasets")
    aisle_label("Aisle 3 · The two suppliers")
    st.markdown("## Two public datasets, integrated at the category level")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div style='padding:1.2rem 1.4rem; background:{PAPER}; border:1px solid rgba(91, 140, 63, 0.18);
            border-radius:14px; box-shadow:0 10px 24px rgba(45, 80, 22, 0.06);'>
<h4 style='margin-top:0;'>Dunnhumby — The Complete Journey</h4>
<b>Used for:</b> price elasticity<br/>
<b>Scale:</b> 2,500 households, 102 weeks, 2.4M product-store-week rows
<div style='margin-top:0.7rem;'><b>Anchor field:</b> regular shelf price equals</div>
<div style='font-family: ui-monospace, Menlo, Consolas, monospace;
     background: #FFF8EC; padding: 0.6rem 0.8rem; border-radius: 4px;
     margin-top: 0.4rem; font-size: 0.85rem; white-space: nowrap; overflow-x: auto;'>
(SALES_VALUE - RETAIL_DISC - COUPON_MATCH_DISC) / QUANTITY
</div>
<div style='margin-top:0.7rem;'><i>Stripping out promotional discounts gives us
the elasticity of base demand to base price.</i></div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div style='padding:1.2rem 1.4rem; background:{PAPER}; border:1px solid rgba(196, 78, 82, 0.16);
            border-radius:14px; box-shadow:0 10px 24px rgba(45, 80, 22, 0.06);'>
<h4 style='margin-top:0;'>Kaggle — Perishable Goods Management</h4>
<b>Used for:</b> demand, spoilage, simulation<br/>
<b>Scale:</b> 100,000 SKU-day rows, 50 stores, 8 categories, 2 years
<div style='margin-top:0.7rem;'><b>Key fields:</b></div>
<div style='font-family: ui-monospace, Menlo, Consolas, monospace;
     background: #FFF8EC; padding: 0.6rem 0.8rem; border-radius: 4px;
     margin-top: 0.4rem; font-size: 0.82rem; line-height: 1.7;'>
days_until_expiry, shelf_life_days, units_wasted,<br/>
spoilage_risk, cost_price, base_price
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### How we connect the two")
    st.markdown("""
The datasets share **no common SKU identifier**. We can't say "this exact
yogurt in Dunnhumby is the same yogurt in the Kaggle data." So the join
happens at the category level. Each perishable category maps to its closest
Dunnhumby department, and the elasticity we learn from Dunnhumby is applied
to the matching category in the perishable optimizer.
""")
    cat_to_dept = pd.DataFrame([
        {"Perishable category": k, "Mapped to department": v}
        for k, v in config.CAT_TO_DEPT.items()
    ])
    st.dataframe(cat_to_dept, width="stretch", hide_index=True)
    st.caption(
        "Pharmaceuticals and Frozen Meals are excluded. Their shelf-life "
        "behaviour doesn't generalise to fresh perishables, and including them "
        "would skew the model. After exclusions, the working dataset has "
        "79,957 records across 8 categories."
    )


def section_showdown():
    divider()
    anchor("showdown")
    aisle_label("Aisle 4 · Strategy showdown")
    st.markdown("## Three pricing rules walk into a 60-day holdout")

    delta = load_csv("strategy_delta_table.csv")
    if delta is None:
        st.warning("strategy_delta_table.csv not found. Run the pipeline first.")
        return

    show = st.multiselect(
        "Compare strategies (toggle to focus)",
        options=delta["Strategy"].tolist(),
        default=delta["Strategy"].tolist(),
        key="showdown_filter",
    )
    df = delta[delta["Strategy"].isin(show)].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="Strategy", y="Profit_USD",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     title="Total profit (60 days, all SKUs)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"${v:,.0f}" for v in df["Profit_USD"]],
                          textposition="outside")
        fig.update_layout(showlegend=False, height=380, yaxis_title="USD")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(df, x="Strategy", y="Waste_Pct",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     title="Waste as % of inventory",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"{v:.1f}%" for v in df["Waste_Pct"]],
                          textposition="outside")
        fig.update_layout(showlegend=False, height=380, yaxis_title="%")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(df, x="Strategy", y="Sell_Through_Pct",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     title="Sell-through rate", template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"{v:.0f}%" for v in df["Sell_Through_Pct"]],
                          textposition="outside")
        fig.update_layout(showlegend=False, height=380, yaxis_title="%")
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = px.bar(df, x="Strategy", y="Revenue_USD",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     title="Total revenue", template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"${v:,.0f}" for v in df["Revenue_USD"]],
                          textposition="outside")
        fig.update_layout(showlegend=False, height=380, yaxis_title="USD")
        st.plotly_chart(fig, width="stretch")

    daily = load_csv("daily_profit_trend.csv")
    if daily is not None:
        daily["date"] = pd.to_datetime(daily["date"])
        long = daily.melt(id_vars="date", var_name="Strategy", value_name="Profit")
        long = long[long["Strategy"].isin(show)]
        fig = px.line(long, x="date", y="Profit", color="Strategy",
                      color_discrete_map=STRAT_PALETTE,
                      title="Daily profit across the 60-day holdout",
                      template=PLOTLY_TEMPLATE)
        fig.update_layout(height=360, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    callout("""
<b>The Fixed 20% number is the most interesting one on this page.</b> A
blanket 20% off actually <i>loses</i> about $70,000 in profit compared to
doing nothing at all. The reason is simple. The 20% rule discounts items
that would have sold at full price anyway, sacrificing margin on those, and
in exchange it only partially helps the items that actually needed help.
<b>Indiscriminate markdowns end up worse than no markdowns.</b> That single
comparison is the strongest argument here for picking discounts at the SKU
level instead of the shelf level.
""", "honest")


def section_aisles():
    divider()
    anchor("aisles")
    aisle_label("Aisle 5 · Aisle by aisle")
    st.markdown("## Where the optimizer wins big, and where it can't help")

    cat = load_csv("category_profit.csv")
    if cat is None:
        st.warning("category_profit.csv not found.")
        return

    cat["uplift"] = cat["Dynamic"] - cat["No Discount"]
    cat = cat.sort_values("uplift", ascending=False)

    pick = st.multiselect(
        "Filter categories",
        options=cat["category"].tolist(),
        default=cat["category"].tolist(),
        key="aisles_filter",
    )
    show = cat[cat["category"].isin(pick)]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=show["category"], y=show["No Discount"],
                              name="No Discount",
                              marker_color=STRAT_PALETTE["No Discount"]))
        fig.add_trace(go.Bar(x=show["category"], y=show["Dynamic"],
                              name="Dynamic",
                              marker_color=STRAT_PALETTE["Dynamic"]))
        fig.update_layout(barmode="group", height=420,
                          title="Profit by category, baseline vs Dynamic",
                          template=PLOTLY_TEMPLATE, yaxis_title="USD")
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = go.Figure(go.Bar(
            x=show["uplift"], y=show["category"], orientation="h",
            marker_color=[LEAF if v > 0 else APPLE for v in show["uplift"]],
            text=[f"${v:+,.0f}" for v in show["uplift"]], textposition="outside",
        ))
        fig.update_layout(height=420, title="Uplift from Dynamic",
                          template=PLOTLY_TEMPLATE, xaxis_title="USD")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(show.set_index("category").round(2), width="stretch")

    callout("""
<b>Honest read on Ready-to-Eat.</b> Even under the Dynamic strategy this
category still loses $158,000. The optimizer pulls it from a $431K loss up
to a $158K loss (so a $272K improvement), but it can't push the category
into the black. The reason is structural. Thin margins meet a high spoilage
rate, and no pricing rule can fix broken unit economics. In a real
deployment you'd either drop this category, renegotiate the supply terms,
or fix shelf-life management upstream before touching the pricing layer.
""", "honest")


def section_expiry():
    divider()
    anchor("expiry")
    aisle_label("Aisle 6 · The sell-by-date game")
    st.markdown("## Discount depth scales with how close the expiry date is")
    st.markdown("""
The most important behaviour we want from the optimizer is simple. If an
item is about to spoil, discount it. If it has plenty of shelf life left,
leave it at full price. Here's whether it actually does that.
""")

    bk = load_csv("near_expiry_buckets.csv")
    if bk is None:
        st.warning("near_expiry_buckets.csv not found.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bk["expiry_bucket"], y=bk["avg_discount_pct"],
        marker_color=APPLE, name="Avg discount",
        text=[f"{v:.1f}%" for v in bk["avg_discount_pct"]],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=bk["expiry_bucket"], y=bk["avg_profit_per_unit"],
        mode="lines+markers", yaxis="y2", name="Profit / unit",
        line=dict(color=LEAF, width=3),
    ))
    fig.update_layout(
        title="Optimizer behaviour by days until expiry",
        template=PLOTLY_TEMPLATE, height=420,
        yaxis=dict(title="Avg discount (%)"),
        yaxis2=dict(title="Profit / unit ($)", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bk.set_index("expiry_bucket"), width="stretch")

    callout("""
<b>Mechanism check passed.</b> Items with one or two days left get an average
23.1% discount, and roughly 76% of them are discounted at all. Items with
six or more days left get nothing. Profit per unit does drop at the deep
end of the discount range, because you're sacrificing margin to convert
soon-to-spoil inventory into actual revenue instead of waste. That's the
trade-off you want the system to make.
""", "good")


def section_register():
    divider()
    anchor("register")
    aisle_label("Aisle 7 · At the register")
    st.markdown("## Most SKUs get no discount at all")

    dist = load_csv("discount_distribution.csv")
    if dist is None:
        st.warning("discount_distribution.csv not found.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(dist, x="discount", y="pct_of_skus",
                     title="Recommended discount distribution (1k-row sample)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color=LEAF,
                          text=[f"{v:.1f}%" for v in dist["pct_of_skus"]],
                          textposition="outside")
        fig.update_layout(height=380, xaxis_tickformat=".0%",
                          xaxis_title="Discount", yaxis_title="% of SKUs")
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("""
**More than half of all SKUs get zero discount.** The optimizer is
conservative by default. It only marks an item down when the combination of
spoilage risk and customer price sensitivity actually justifies the lost
margin.

That's the biggest design difference between this and a flat 20% rule. The
optimizer's discounts are earned, not sprayed across the shelf.
""")

    cat_disc = load_csv("category_mean_discount.csv")
    if cat_disc is not None:
        st.subheader("Mean recommended discount by category")
        fig = px.bar(cat_disc.sort_values("discount", ascending=False),
                     x="category", y="discount",
                     color="discount", color_continuous_scale=[CREAM, APPLE],
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(height=360, yaxis_title="Avg discount (%)",
                          showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


def section_stress():
    divider()
    anchor("stress")
    aisle_label("Aisle 8 · Near-expiry stress test")
    st.markdown("## When everything is about to spoil at once")

    hs = load_csv("high_spoilage_kpi.csv")
    if hs is None:
        st.warning("high_spoilage_kpi.csv not found.")
        return

    if hs.columns[0] != "strategy":
        hs = hs.rename(columns={hs.columns[0]: "strategy"})
    hs["label"] = hs["strategy"].map({"no_discount": "No Discount",
                                       "fixed_20": "Fixed 20%",
                                       "dynamic": "Dynamic"})

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(hs, x="label", y="Total_Profit", color="label",
                     color_discrete_map=STRAT_PALETTE,
                     title="Profit on near-expiry subset (<=3 days to expiry)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"${v:,.0f}" for v in hs["Total_Profit"]],
                          textposition="outside")
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(hs, x="label", y="Total_Waste", color="label",
                     color_discrete_map=STRAT_PALETTE,
                     title="Units wasted on near-expiry subset",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(text=[f"{v:,.0f}" for v in hs["Total_Waste"]],
                          textposition="outside")
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(hs.set_index("label").drop(columns=["strategy"]),
                 width="stretch")

    callout("""
<b>Every strategy is unprofitable on this slice.</b> Heavy markdown plus
residual waste eats margin no matter what you do. But the relative gain
from the Dynamic strategy is biggest right here. It cuts losses by roughly
$1.1M versus the no-discount baseline on the near-expiry items alone. The
case for dynamic pricing is strongest exactly where the alternatives hurt
the most.
""", "warn")


def section_sensitivity():
    divider()
    anchor("sensitivity")
    aisle_label("Aisle 9 · What if we got elasticity wrong?")
    st.markdown("## Sensitivity analysis, and why the pretty version is misleading")

    es = load_csv("elasticity_sensitivity.csv")
    if es is None:
        st.warning("elasticity_sensitivity.csv not found.")
        return

    overall = es[es["discount"] == -1.0].copy()
    overall["scenario"] = pd.Categorical(
        overall["scenario"],
        categories=["Elast x 0.5", "Baseline", "Elast x 1.5"], ordered=True,
    )
    overall = overall.sort_values(["subset", "scenario"])

    subsets = overall["subset"].unique().tolist()
    cols = st.columns(len(subsets))
    for col, sub in zip(cols, subsets):
        with col:
            sub_df = overall[overall["subset"] == sub]
            label = ("Full sample, dominated by near-expiry rows that hit the override"
                     if sub == "ALL_ROWS"
                     else "Non-near-expiry rows only, which is the real test")
            fig = px.bar(sub_df, x="scenario", y="mean_profit",
                         title=label, color="scenario",
                         color_discrete_sequence=["#A0A0A0", LEAF, "#A0A0A0"],
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(text=[f"${v:.2f}" for v in sub_df["mean_profit"]],
                              textposition="outside")
            fig.update_layout(height=400, showlegend=False, xaxis_title="")
            st.plotly_chart(fig, width="stretch")

    callout("""
<b>Read this one carefully.</b> On the full sample, mean profit barely
moves across 0.5x, 1.0x, and 1.5x elasticity. That looks like the optimizer
is robust to elasticity, but it isn't. The reason is that <b>42% of holdout
rows are near-expiry</b>, and those rows use the fixed -2.0 elasticity
override regardless of the multiplier we tweak.<br/><br/>
The honest test is the <b>non-near-expiry subset</b> on the right. If that
also barely moves under plus or minus 50%, then the regular elasticity is
doing limited work in the optimizer's decisions. That's a real finding
worth reporting, not the same thing as "stability." Don't oversell this
section in the writeup.
""", "honest")

    st.subheader("Discount distribution under each scenario")
    disc = es[es["discount"] >= 0].copy()
    disc["discount_pct"] = (disc["discount"] * 100).astype(int)
    fig = px.bar(
        disc, x="discount_pct", y="n_rows",
        color="scenario", facet_col="subset", barmode="group",
        labels={"discount_pct": "Discount (%)", "n_rows": "# of SKUs"},
        title="How elasticity changes shift the discount distribution",
        color_discrete_sequence=[STRAT_PALETTE["No Discount"], LEAF, APPLE],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")


def section_whatif():
    divider()
    anchor("whatif")
    aisle_label("Aisle 10 · Try it yourself")
    st.markdown("## Move the dials and re-run the optimizer with your assumptions")

    sample = load_csv("whatif_sample.csv")
    bundle = load_spoilage_model()
    fc_df = load_csv("sku_demand_forecasts.csv")

    if sample is None or bundle is None:
        st.error("""
The What-if explorer needs **`outputs/whatif_sample.csv`** and
**`models/spoilage_model.pkl`**. Run `python save_whatif_files.py` from the
`code/` folder, then refresh this page.
""")
        return

    spoilage_model = bundle["model"]
    train_cols = bundle["train_cols"]
    waste_frac = bundle["waste_frac"]

    if fc_df is not None:
        fc_df["date"] = pd.to_datetime(fc_df["date"])
        forecast_lookup = {(r["date"], r["category"]): r["forecast"]
                           for _, r in fc_df.iterrows()}
    else:
        forecast_lookup = {}

    sample = sample.copy()
    sample["transaction_date"] = pd.to_datetime(sample["transaction_date"])

    st.markdown("### Your assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        elas_mult = st.slider(
            "Regular elasticity multiplier", 0.25, 2.0, 1.0, 0.25,
            help="Scales the per-department elasticity. The near-expiry override is unaffected.",
        )
    with c2:
        ne_thresh = st.slider(
            "Near-expiry threshold (days)", 1, 14, config.NEAR_EXPIRY_DAYS, 1,
            help="Items at or below this many days to expiry use the bargain-hunter elasticity.",
        )
    with c3:
        ne_elas = st.slider(
            "Near-expiry elasticity", -4.0, -1.0, config.NEAR_EXPIRY_ELAS, 0.25,
            help="Literature: -1.5 to -3.0. The default is -2.0.",
        )
    c4, c5 = st.columns(2)
    with c4:
        margin = st.slider("Margin floor (price >= cost x)",
                           1.0, 1.20, 1.0 + config.MIN_MARGIN, 0.01)
    with c5:
        max_disc = st.slider("Maximum allowed discount (%)",
                             10, 60, 40, 10)
    grid = [round(d / 100, 2) for d in range(0, max_disc + 1, 10)]
    st.caption(f"Discount grid: {[f'{int(d * 100)}%' for d in grid]}")

    if st.button("▶ Re-run optimizer with these settings", type="primary"):
        with st.spinner("Re-running optimizer on the 500-row sample..."):
            from optimizer import batch_simulate

            base_elas = config.DEFAULT_ELASTICITY_MAP
            scaled = {k: v * elas_mult for k, v in base_elas.items()}

            old_grid = config.DISCOUNT_GRID
            old_thresh = config.NEAR_EXPIRY_DAYS
            old_ne_elas = config.NEAR_EXPIRY_ELAS
            old_margin = config.MIN_MARGIN
            try:
                config.DISCOUNT_GRID = grid
                config.NEAR_EXPIRY_DAYS = ne_thresh
                config.NEAR_EXPIRY_ELAS = ne_elas
                config.MIN_MARGIN = margin - 1.0
                user_run = batch_simulate(
                    sample, "dynamic",
                    spoilage_model=spoilage_model, train_cols=train_cols,
                    elasticity_map=scaled, waste_frac=waste_frac,
                    demand_forecasts=forecast_lookup,
                )
                config.DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
                config.NEAR_EXPIRY_DAYS = 5
                config.NEAR_EXPIRY_ELAS = -2.0
                config.MIN_MARGIN = 0.05
                base_run = batch_simulate(
                    sample, "dynamic",
                    spoilage_model=spoilage_model, train_cols=train_cols,
                    elasticity_map=base_elas, waste_frac=waste_frac,
                    demand_forecasts=forecast_lookup,
                )
            finally:
                config.DISCOUNT_GRID = old_grid
                config.NEAR_EXPIRY_DAYS = old_thresh
                config.NEAR_EXPIRY_ELAS = old_ne_elas
                config.MIN_MARGIN = old_margin

        c1, c2, c3 = st.columns(3)
        c1.metric("Profit (your settings)",
                  f"${user_run['profit'].sum():,.0f}",
                  delta=f"${user_run['profit'].sum() - base_run['profit'].sum():,.0f} vs project baseline")
        c2.metric("Expected waste (units)",
                  f"{user_run['exp_waste'].sum():,.0f}",
                  delta=f"{user_run['exp_waste'].sum() - base_run['exp_waste'].sum():,.0f}",
                  delta_color="inverse")
        c3.metric("Avg discount applied",
                  f"{user_run['discount'].mean() * 100:.1f}%",
                  delta=f"{(user_run['discount'].mean() - base_run['discount'].mean()) * 100:+.1f} pp")

        st.subheader("Discount distribution — your settings vs the project baseline")
        comp = pd.concat([
            user_run.assign(scenario="Your settings"),
            base_run.assign(scenario="Project baseline"),
        ])
        comp["discount_pct"] = (comp["discount"] * 100).round().astype(int)
        cnt = comp.groupby(["scenario", "discount_pct"]).size().reset_index(name="n")
        fig = px.bar(cnt, x="discount_pct", y="n", color="scenario", barmode="group",
                     labels={"discount_pct": "Discount (%)", "n": "# of SKUs"},
                     color_discrete_sequence=[LEAF, STRAT_PALETTE["No Discount"]],
                     template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width="stretch")


def section_models():
    divider()
    anchor("models")
    aisle_label("Aisle 11 · Under the hood")
    st.markdown("## Model selection details")
    st.markdown(
        "Click any panel below to expand it. These are the comparison tables "
        "that drove the model picks."
    )

    with st.expander("Demand forecasting — 30-day validation comparison"):
        cmp_demand = load_csv("demand_model_comparison.csv")
        if cmp_demand is not None:
            st.dataframe(cmp_demand, width="stretch", hide_index=True)
            st.caption("Selection rule: highest directional accuracy, RMSE tiebreak.")
        else:
            st.warning("demand_model_comparison.csv not found.")

    with st.expander("Spoilage classifier — 4-way comparison"):
        cmp_spoil = load_csv("spoilage_model_comparison.csv")
        if cmp_spoil is not None:
            st.dataframe(cmp_spoil, width="stretch", hide_index=True)
            st.caption("Selection rule: highest ROC-AUC; Brier tiebreak within 0.005 AUC. "
                       "Brier matters because the optimizer uses raw probabilities.")
        else:
            st.warning("spoilage_model_comparison.csv not found.")

    with st.expander("Department elasticities — out-of-sample validation"):
        elas = load_csv("elasticity_estimates.csv")
        if elas is not None:
            st.dataframe(elas, width="stretch", hide_index=True)
            st.caption(
                "A negative OOS R squared looks alarming but it isn't a broken "
                "model. The out-of-sample test deliberately strips out the "
                "store and week fixed effects (those are unknown for future "
                "weeks), leaving only price as the predictor. What we're "
                "validating is whether the elasticity coefficient itself is "
                "stable across time windows. It is."
            )
        else:
            st.warning("elasticity_estimates.csv not found.")


def section_limits():
    divider()
    anchor("limits")
    aisle_label("Aisle 12 · The honest limits")
    st.markdown("## What this analysis doesn't claim")

    st.markdown("""
A capstone is judged as much on what it doesn't claim as on what it does.
There are five real holes in this analysis. We're calling them out plainly
instead of burying them.
""")

    items = [
        ("1.  Elasticity transfer",
         "The elasticities we use come from Dunnhumby, which captures normal "
         "everyday grocery shopping. But a customer responding to a 30% off "
         "sticker on something that expires tomorrow is making a different "
         "kind of decision than one filling their weekly basket. We handle "
         "this with a -2.0 near-expiry override, informed by published "
         "markdown literature. That number is plausible, but **it isn't "
         "estimated from our own data**. A real deployment would need a "
         "small A/B markdown experiment to calibrate it properly."),
        ("2.  Ready-to-Eat is structurally negative",
         "Even under Dynamic pricing this category loses $158,000. The "
         "optimizer recovers $272,000 versus the no-discount baseline, but "
         "it can't push the category into the black. The right reading "
         "isn't that the model failed. It's that the unit economics are "
         "broken. The fix has to be upstream: margin renegotiation, cold "
         "chain, or shelf-life management. Pricing alone can't get there."),
        ("3.  The spoilage target is binary",
         "We predict whether any waste happens, not how much. We patch "
         "this by multiplying the predicted probability by an empirical "
         "waste fraction (0.1727) to translate it into expected wasted "
         "units. That's a reasonable workaround, but for high-inventory "
         "SKUs the difference between 5 wasted units and 200 is enormous. "
         "A regression model that predicts the waste fraction directly "
         "would be more precise. Left for future work."),
        ("4.  The elasticity sensitivity test looks too clean",
         "Under plus or minus 50% elasticity on the full sample, the "
         "optimizer's profit barely moves. That's mostly mechanical: 42% "
         "of rows hit the -2.0 override which doesn't scale with the "
         "multiplier. The honest test is on the non-near-expiry subset, "
         "and we show both. The report should not present the full-sample "
         "result as proof of stability."),
        ("5.  Demand is a point forecast, not probabilistic",
         "The optimizer treats the LightGBM forecast as a known number. "
         "In a real deployment, forecast uncertainty matters: a high-"
         "variance forecast should pull the optimizer toward more "
         "conservative discounts. We don't model that here."),
    ]
    for title, body in items:
        st.markdown(f"#### {title}")
        st.markdown(body)


def section_checkout():
    divider()
    anchor("checkout")
    aisle_label("Aisle 13 · The final receipt")
    st.markdown("## Checkout: what we showed and what comes next")

    delta = load_csv("strategy_delta_table.csv")

    st.markdown("""
### What the simulation showed

On a genuine 60-day out-of-sample holdout, the per-SKU per-day pricing
optimizer (demand forecast plus department elasticity plus calibrated
spoilage probability):

- Delivered roughly **$2M in profit, versus $420K under the no-discount
  baseline**. About 4.8 times more.
- Cut expected waste roughly **in half**.
- Raised sell-through from **66% to 79%**.

The most important single number on the page is actually the Fixed 20%
result: **a flat 20% discount underperforms doing nothing at all** ($349K
versus $420K). That comparison is the strongest argument here for picking
discounts at the SKU level instead of spraying them across the shelf.

### What we didn't show

A few things that would matter for a real deployment but aren't in this
project:

- Whether these gains hold in a real store. This is a simulation. The next
  step is a small field pilot in a handful of stores.
- Whether the 17.27% empirical waste fraction generalises across store
  formats, regions, or seasons.
- Whether the -2.0 near-expiry elasticity is right. The literature puts it
  somewhere between -1.5 and -3.0, and we picked the midpoint.

### What a real deployment would look like

A daily batch job. For every SKU in every store: pull yesterday's
inventory and remaining shelf life, call the demand forecaster, query the
spoilage model at each candidate price, run the discount grid, output one
recommended markdown. Add operational guardrails on top, like a cap on the
fraction of SKUs that can be discounted at once, or executive override on
premium items. Then A/B test the new policy against the current ad-hoc
markdown rules on a handful of stores before rolling it out chain-wide.
""")

    if delta is not None:
        st.subheader("Final scoreboard")
        st.dataframe(delta.set_index("Strategy"), width="stretch")

    st.markdown("---")
    st.markdown(f"""
<div style='text-align:center; color:{BAG}; font-style:italic; padding:1.5rem;'>
FSE 570 Capstone · Team Missouri<br/>
Bhavana Reddy Pasula · Sneha Nannapaneni · Moksha Smruthi Morapakula ·<br/>
Sree Padma Priya Abburi · Srujana Rachamalla
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main — render everything top-to-bottom
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    # Inject scroll-tracking JS for the journey line
    st.markdown(JOURNEY_JS, unsafe_allow_html=True)
    section_welcome()
    section_problem()
    section_approach()
    section_datasets()
    section_showdown()
    section_aisles()
    section_expiry()
    section_register()
    section_stress()
    section_sensitivity()
    section_whatif()
    section_models()
    section_limits()
    section_checkout()


if __name__ == "__main__":
    main()
