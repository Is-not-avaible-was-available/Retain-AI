# RETAIN-AI
# AI-Powered Customer Retention Decision Platform
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import html
import altair as alt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = DATA_DIR / "features.parquet"

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retain-AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
    <style>


    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #F5F7FA;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0B1220;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #0B1220;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1 !important;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        color: #0F172A !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px;
    }

    h2 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    h3 {
        color: #1E293B !important;
        font-weight: 650 !important;
    }

    p {
        color: #475569;
    }


    /* ========================================================
       CUSTOM CARDS
       ======================================================== */

    .retain-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .retain-card-title {
        color: #0F172A;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .retain-card-subtitle {
        color: #64748B;
        font-size: 0.80rem;
        line-height: 1.5;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        min-height: 118px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .kpi-label {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        margin-bottom: 9px;
    }

    .kpi-value {
        color: #0F172A;
        font-size: 1.65rem;
        font-weight: 750;
        line-height: 1.15;
    }

    .kpi-subtitle {
        color: #94A3B8;
        font-size: 0.73rem;
        margin-top: 7px;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand-name {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.75rem;
        line-height: 1.45;
        margin-top: 5px;
    }

    .sidebar-label {
        color: #64748B;
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 24px;
        margin-bottom: 8px;
    }

    .sidebar-stat {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 11px 13px;
        margin-bottom: 8px;
    }

    .sidebar-stat-label {
        color: #64748B;
        font-size: 0.64rem;
        font-weight: 650;
    }

    .sidebar-stat-value {
        color: #F8FAFC;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 3px;
    }

    .sidebar-info {
        color: #94A3B8;
        font-size: 0.72rem;
        line-height: 1.55;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 26px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .hero-title {
        color: #0F172A;
        font-size: 1.75rem;
        font-weight: 750;
        letter-spacing: -0.7px;
    }

    .hero-subtitle {
        color: #64748B;
        font-size: 0.88rem;
        margin-top: 5px;
    }

    .hero-date {
        color: #94A3B8;
        font-size: 0.73rem;
        margin-top: 12px;
    }


    /* ========================================================
       INSIGHT BOX
       ======================================================== */

    .insight-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 17px 19px;
    }

    .insight-label {
        color: #64748B;
        font-size: 0.69rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .insight-text {
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.65;
        margin-top: 7px;
    }


    /* ========================================================
       RISK BADGES
       ======================================================== */

    .risk-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .risk-critical {
        background: #FEE2E2;
        color: #991B1B;
    }

    .risk-high {
        background: #FFEDD5;
        color: #9A3412;
    }

    .risk-moderate {
        background: #FEF3C7;
        color: #92400E;
    }

    .risk-low {
        background: #DCFCE7;
        color: #166534;
    }


    /* ========================================================
       SIGNAL CARDS
       ======================================================== */

    .signal-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        min-height: 245px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .signal-title {
        color: #0F172A;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .signal-subtitle {
        color: #64748B;
        font-size: 0.76rem;
        margin-top: 3px;
        margin-bottom: 15px;
    }

    .signal-row {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 9px;
        padding: 10px 12px;
        margin-bottom: 7px;
    }

    .signal-label {
        color: #64748B;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .signal-value {
        color: #0F172A;
        font-size: 0.90rem;
        font-weight: 700;
        margin-top: 2px;
    }


            /* =========================================================
           CUSTOMER 360 — SHAP
           ========================================================= */
        
        .shap-section {
            margin-top: 10px;
        }
        
        .shap-column {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-radius: 16px;
            padding: 22px;
            min-height: 100%;
        }
        
        .shap-column-title {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 18px;
            font-weight: 700;
            color: #14213d;
            margin-bottom: 20px;
        }
        
        .shap-icon-risk {
            color: #e54868;
            font-size: 19px;
        }
        
        .shap-icon-safe {
            color: #3b82f6;
            font-size: 19px;
        }
        
        .shap-driver {
            padding: 15px 0 17px 0;
            border-bottom: 1px solid #edf0f4;
        }
        
        .shap-driver:last-child {
            border-bottom: none;
        }
        
        .shap-driver-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        
        .shap-feature {
            font-size: 14px;
            font-weight: 650;
            color: #253858;
        }
        
        .shap-value {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            font-weight: 600;
            color: #344054;
            background: #f6f8fa;
            border: 1px solid #e5e9ef;
            padding: 4px 8px;
            border-radius: 6px;
            white-space: nowrap;
        }
        
        .shap-driver-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            margin-bottom: 8px;
            font-size: 12px;
            color: #8a94a6;
        }
        
        .shap-driver-meta span:first-child {
            color: #4b5565;
            font-weight: 600;
        }
        
        .shap-track {
            width: 100%;
            height: 6px;
            background: #edf1f5;
            border-radius: 99px;
            overflow: hidden;
        }
        
        .shap-bar-positive,
        .shap-bar-negative {
            height: 100%;
            border-radius: 99px;
        }
        
        .shap-explanation-note {
            margin-top: 14px;
            font-size: 12px;
            color: #8a94a6;
        }

    /* ========================================================
       MODEL & GOVERNANCE
       ======================================================== */

    .governance-card {
        background: #FFFFFF;
        border: 1px solid #D8E3F0;
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 150px;
        box-shadow: 0 5px 18px rgba(30,64,110,0.045);
    }

    .governance-kicker {
        color: #607795;
        font-size: 0.67rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }

    .governance-value {
        color: #102A56;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .governance-note {
        color: #6D83A2;
        font-size: 0.74rem;
        line-height: 1.5;
        margin-top: 6px;
    }

    .governance-panel {
        background: linear-gradient(120deg, #F7FAFF 0%, #F4F1FF 100%);
        border: 1px solid #D7E3F2;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(30,64,110,0.045);
    }

    .governance-panel-title {
        color: #17345D;
        font-size: 0.86rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .governance-panel-text {
        color: #405674;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    .architecture-flow {
        display: flex;
        align-items: stretch;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .architecture-node {
        flex: 1 1 145px;
        min-width: 125px;
        background: #FFFFFF;
        border: 1px solid #D7E3F2;
        border-radius: 11px;
        padding: 13px 12px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(30,64,110,0.035);
    }

    .architecture-node strong {
        display: block;
        color: #17345D;
        font-size: 0.77rem;
        line-height: 1.3;
    }

    .architecture-node span {
        display: block;
        color: #71829A;
        font-size: 0.66rem;
        line-height: 1.4;
        margin-top: 4px;
    }

    .architecture-arrow {
        display: flex;
        align-items: center;
        color: #6B7FA2;
        font-size: 1rem;
        font-weight: 800;
    }

    .governance-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        background: #FFFFFF;
        border: 1px solid #D8E3F0;
        border-radius: 12px;
        overflow: hidden;
    }

    .governance-table th {
        text-align: left;
        color: #526F9A;
        background: #F4F8FD;
        padding: 10px 12px;
        font-weight: 800;
        border-bottom: 1px solid #DCE7F3;
    }

    .governance-table td {
        color: #334155;
        padding: 10px 12px;
        border-bottom: 1px solid #EDF2F7;
        vertical-align: top;
        line-height: 1.45;
    }

    .governance-table tr:last-child td {
        border-bottom: none;
    }

    .governance-status {
        display: inline-flex;
        align-items: center;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 0.67rem;
        font-weight: 800;
    }

    .governance-status-ready {
        background: #DCFCE7;
        color: #166534;
    }

    .governance-status-monitor {
        background: #FEF3C7;
        color: #92400E;
    }

    .governance-disclaimer {
        background: #FFF8E8;
        border: 1px solid #F3D99A;
        border-radius: 12px;
        padding: 13px 15px;
        color: #72520E;
        font-size: 0.75rem;
        line-height: 1.55;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.70rem;
        line-height: 1.6;
        padding-top: 35px;
    }

    /* ========================================================
       CUSTOMER 360 — AI RETENTION INSIGHT
       ======================================================== */

    .ai-insight-card {
        background: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }

    .ai-insight-kicker {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .ai-insight-title {
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 4px;
    }

    .ai-insight-summary {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 12px;
    }

    .ai-insight-label {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 7px;
    }

    /* ========================================================
       EXECUTIVE AI
       ======================================================== */

    .executive-ai-card {
        background: linear-gradient(135deg, #F7FAFF 0%, #F3F0FF 100%);
        border: 1px solid #D5E1F2;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(30,64,110,0.07);
    }

    .executive-ai-kicker {
        color: #315EF7;
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .executive-ai-title {
        color: #102A56;
        font-size: 1.28rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .executive-ai-summary {
        color: #29405F;
        font-size: 0.96rem;
        line-height: 1.65;
        margin-top: 10px;
    }

    .executive-ai-status {
        display: inline-flex;
        align-items: center;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 800;
        margin-top: 13px;
    }

    .executive-ai-status-stable {
        background: #DCFCE7;
        color: #166534;
    }

    .executive-ai-status-watch {
        background: #FEF3C7;
        color: #92400E;
    }

    .executive-ai-status-elevated {
        background: #FFEDD5;
        color: #9A3412;
    }

    .executive-ai-status-critical {
        background: #FEE2E2;
        color: #991B1B;
    }

    .executive-ai-section {
        background: #FFFFFF;
        border: 1px solid #DCE6F2;
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 190px;
        box-shadow: 0 4px 14px rgba(30,64,110,0.045);
    }

    .executive-ai-section-title {
        color: #31527D;
        font-size: 0.70rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        margin-bottom: 9px;
    }

    .executive-ai-section ul {
        margin: 0 0 0 17px;
        padding: 0;
        color: #334155;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    .executive-ai-footer {
        color: #71829A;
        font-size: 0.70rem;
        line-height: 1.55;
        margin-top: 12px;
    }


    /* ========================================================
       RETAIN-AI COLOR SYSTEM
       ======================================================== */

    .stApp {
        background: linear-gradient(135deg, #F4F7FC 0%, #EEF4FB 55%, #F7F3FC 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1730 0%, #111F3D 55%, #17284B 100%) !important;
        border-right: 1px solid #243B68 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] * {
        color: #E8EEF9;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #C9D5E8 !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #FFFFFF !important;
    }

    .sidebar-stat {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 8px 24px rgba(3,12,30,0.18);
    }

    .sidebar-label {
        color: #8EA8CF;
    }

    .sidebar-stat-label {
        color: #91A6C7;
    }

    .sidebar-stat-value {
        color: #F7FAFF;
    }

    .sidebar-info {
        color: #AFC0D9;
    }

    .hero {
        background: linear-gradient(115deg, #FFFFFF 0%, #F8FBFF 60%, #F4F1FF 100%);
        border: 1px solid #D8E3F2;
        box-shadow: 0 10px 30px rgba(37,99,235,0.07);
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -50px;
        top: -80px;
        background: radial-gradient(circle, rgba(99,102,241,0.14), rgba(99,102,241,0));
        pointer-events: none;
    }

    .hero-title {
        color: #102A56;
    }

    .hero-subtitle {
        color: #547096;
    }

    .hero-date {
        color: #7890B2;
    }

    .retain-card,
    .kpi-card,
    .signal-card {
        border-color: #D8E3F0;
        box-shadow: 0 5px 18px rgba(30,64,110,0.055);
    }

    .kpi-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #FBFDFF 100%);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #2563EB, #7C3AED);
        border-radius: 4px 0 0 4px;
    }

    .kpi-label {
        color: #5B7192;
    }

    .kpi-value {
        color: #102A56;
    }

    .kpi-subtitle {
        color: #8297B5;
    }

    .insight-box {
        background: linear-gradient(120deg, #EEF5FF, #F5F1FF);
        border: 1px solid #D5E3F5;
    }

    .insight-label {
        color: #526F9A;
    }

    .insight-text {
        color: #29405F;
    }

    .signal-row {
        background: #F4F8FD;
        border-color: #DCE7F3;
    }

    .signal-title {
        color: #102A56;
    }

    .signal-subtitle {
        color: #6D83A2;
    }

    .signal-label {
        color: #6D83A2;
    }

    .signal-value {
        color: #17345D;
    }

    /* Streamlit controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-color: #C9D8EA !important;
        background: #FFFFFF !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #5B7CFA !important;
        box-shadow: 0 0 0 1px #5B7CFA !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: #4F6FF5 !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 2px 7px rgba(79,111,245,0.35);
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        border-radius: 999px;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #315EF7 0%, #6546D9 100%) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        box-shadow: 0 5px 14px rgba(49,94,247,0.20);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #244FE2 0%, #5839C8 100%) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        box-shadow: 0 7px 18px rgba(49,94,247,0.28);
    }

    .stDownloadButton > button {
        padding: 0.62rem 1rem !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #D8E3F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 18px rgba(30,64,110,0.04);
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #CFE0F5;
    }

    /* Charts */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #D8E3F0;
        border-radius: 14px;
        padding: 16px 18px 10px 18px;
        box-shadow: 0 5px 18px rgba(30,64,110,0.045);
    }

    
    /* ========================================================
       READABILITY + VISUAL POLISH
       ======================================================== */

    /* Stronger page hierarchy */
    .main .block-container > div:first-child {
        scroll-margin-top: 20px;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
        color: #102A56 !important;
    }

    h2 {
        font-size: 1.35rem !important;
        color: #183B67 !important;
    }

    h3 {
        font-size: 1.08rem !important;
        color: #244B78 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #526784 !important;
    }

    /* Streamlit metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8FBFF 100%) !important;
        border: 1px solid #D8E3F0 !important;
        border-radius: 14px !important;
        padding: 15px 17px !important;
        box-shadow: 0 5px 18px rgba(30,64,110,0.055) !important;
        min-height: 92px !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #607795 !important;
        font-size: 0.72rem !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #102A56 !important;
        font-weight: 800 !important;
        font-size: 1.55rem !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.70rem !important;
        font-weight: 650 !important;
    }

    /* Give native Streamlit containers a clean light surface */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #D8E3F0 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #17345D !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #607795 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #315EF7 !important;
    }

    /* Inputs and labels */
    label[data-testid="stWidgetLabel"] p,
    .stTextInput label,
    .stMultiSelect label,
    .stSelectSlider label {
        color: #294B78 !important;
        font-weight: 700 !important;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        background: #FFFFFF !important;
    }

    /* Charts: force a light, readable surface even when the browser/app
       is using a dark system theme. */
    .chart-card {
        background: #FFFFFF !important;
        border: 1px solid #D7E3F2 !important;
        border-radius: 16px !important;
        padding: 16px 18px 12px 18px !important;
        box-shadow: 0 7px 22px rgba(30,64,110,0.06) !important;
        overflow: hidden !important;
    }

    .chart-card .stCaption,
    .chart-card [data-testid="stCaptionContainer"] {
        color: #405674 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stVegaLiteChart"],
    .stVegaLiteChart,
    .vega-embed {
        background: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Small visual section separators */
    hr {
        border: 0 !important;
        border-top: 1px solid #DCE7F3 !important;
        margin: 1.2rem 0 !important;
    }

    /* Coloured status chips */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 750;
        border: 1px solid transparent;
    }

    .status-chip-blue {
        background: #EAF1FF;
        color: #2454C6;
        border-color: #C9DAFF;
    }

    .status-chip-purple {
        background: #F1EBFF;
        color: #6D35C7;
        border-color: #DDD0FF;
    }

    .status-chip-teal {
        background: #E7FAF6;
        color: #087F70;
        border-color: #BDEDE5;
    }

    /* Better alert readability */
    div[data-testid="stAlert"] p {
        color: #29405F !important;
    }

    /* Mobile safety */
    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-title {
            font-size: 1.45rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
    }

</style>
    """
)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_currency(value):

    if pd.isna(value):
        return "₹0"

    value = float(value)

    if abs(value) >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"

    if abs(value) >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} L"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"

    return f"₹{value:,.0f}"


def format_probability(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.1f}%"


def format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.0f}"


def risk_tier(probability):

    if probability >= 0.20:
        return "Critical"

    if probability >= 0.10:
        return "High"

    if probability >= 0.05:
        return "Moderate"

    return "Low"


def exposure_tier(value, thresholds):

    if value >= thresholds["critical"]:
        return "Critical"

    if value >= thresholds["high"]:
        return "High"

    if value >= thresholds["moderate"]:
        return "Moderate"

    return "Low"


def renewal_urgency(days):

    if pd.isna(days):
        return "Unknown"

    days = float(days)

    if days <= 30:
        return "Critical"

    if days <= 90:
        return "High"

    if days <= 180:
        return "Moderate"

    if days <= 365:
        return "Low"

    return "Very Low"


def health_signal(row):

    health = row.get(
        "health_score",
        np.nan
    )

    trend = row.get(
        "health_trend",
        np.nan
    )

    behaviour = row.get(
        "behaviour",
        ""
    )

    if (
        (pd.notna(health) and health < 40)
        or
        (pd.notna(trend) and trend < -0.20)
        or
        behaviour == "Rapidly Declining"
    ):
        return "Account Health Intervention"

    if (
        (pd.notna(health) and health < 60)
        or
        (pd.notna(trend) and trend < -0.05)
        or
        behaviour == "Declining"
    ):
        return "Customer Success Monitoring"

    return "Healthy"


def usage_signal(row):

    severe = row.get(
        "severe_usage_decline_12w_flag",
        0
    )

    decline_12 = row.get(
        "usage_decline_12w_flag",
        0
    )

    decline_4 = row.get(
        "usage_decline_4w_flag",
        0
    )

    if severe == 1:
        return "Severe Product Engagement Risk"

    if decline_12 == 1:
        return "Product Adoption Intervention"

    if decline_4 == 1:
        return "Early Engagement Warning"

    return "Stable Usage"


def support_signal(row):

    high_burden = row.get(
        "high_support_burden_12w",
        0
    )

    high_escalation = row.get(
        "high_escalation_12w",
        0
    )

    low_csat = row.get(
        "low_csat_12w",
        0
    )

    slow_resolution = row.get(
        "slow_resolution_12w",
        0
    )

    if high_burden == 1 or high_escalation == 1:
        return "Technical / CX Escalation"

    if low_csat == 1 or slow_resolution == 1:
        return "Customer Experience Review"

    return "Normal Support Profile"


def recommended_action(row):

    actions = []

    health = row.get(
        "health_score",
        np.nan
    )

    trend = row.get(
        "health_trend",
        np.nan
    )

    behaviour = row.get(
        "behaviour",
        ""
    )

    if (
        (pd.notna(health) and health < 40)
        or
        (pd.notna(trend) and trend < -0.20)
        or
        behaviour == "Rapidly Declining"
    ):
        actions.append(
            "Account Health Review"
        )

    if (
        row.get(
            "severe_usage_decline_12w_flag",
            0
        ) == 1
        or
        row.get(
            "usage_decline_12w_flag",
            0
        ) == 1
    ):
        actions.append(
            "Product Adoption"
        )

    if (
        row.get(
            "high_support_burden_12w",
            0
        ) == 1
        or
        row.get(
            "high_escalation_12w",
            0
        ) == 1
    ):
        actions.append(
            "CX / Technical Escalation"
        )

    days = row.get(
        "days_to_renewal",
        np.nan
    )

    if pd.notna(days) and days <= 90:
        actions.append(
            "Renewal Engagement"
        )

    if not actions:
        actions.append(
            "Proactive Customer Monitoring"
        )

    return " • ".join(actions)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(
    show_spinner="Loading feature store..."
)
def load_features():

    if not FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"""
Feature store not found.

Expected location:
{FEATURES_PATH}
"""
        )

    df = pd.read_parquet(
        FEATURES_PATH
    )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    return df


@st.cache_data(
    show_spinner="Preparing portfolio..."
)
def prepare_portfolio(features):

    latest_date = (
        features[
            "snapshot_date"
        ].max()
    )

    portfolio = (
        features[
            features[
                "snapshot_date"
            ] == latest_date
        ]
        .copy()
        .reset_index(drop=True)
    )

    # IMPORTANT:
    # Do not expose actual churn labels
    # in the production/dashboard layer.

    portfolio = portfolio.drop(
        columns=[
            "churn_90d"
        ],
        errors="ignore"
    )

    return portfolio, latest_date


# ============================================================
# MODEL INFERENCE
# ============================================================

@st.cache_data(
    show_spinner="Running calibrated XGBoost..."
)
def generate_predictions(portfolio):

    from inference import (
        predict_churn_probability
    )

    probabilities = (
        predict_churn_probability(
            portfolio
        )
    )

    result = portfolio.copy()

    result[
        "churn_probability"
    ] = probabilities

    return result


@st.cache_data(
    show_spinner="Building portfolio history from the production model..."
)
def generate_historical_portfolio_predictions(features):
    """
    Run the frozen production-calibrated XGBoost model across historical
    feature-store snapshots for portfolio trend analysis.

    Inference is performed one weekly snapshot at a time so the deployment
    process never creates a second full-size copy of the feature store.
    Only the compact six-column historical result is retained.
    """

    import gc

    from inference import predict_churn_probability

    historical_results = []

    # IMPORTANT: never copy the entire 424k-row feature store here.
    # Process one snapshot at a time to keep peak RAM low on Streamlit Cloud.
    for snapshot_date, snapshot_group in features.groupby(
        "snapshot_date",
        sort=True,
    ):
        snapshot = snapshot_group.drop(
            columns=["churn_90d"],
            errors="ignore",
        ).copy().reset_index(drop=True)

        probabilities = predict_churn_probability(snapshot)

        annual_contract_value = (
            snapshot["annual_contract_value"]
            .fillna(0)
            .to_numpy()
        )

        historical_results.append(
            pd.DataFrame(
                {
                    "customer_id": snapshot["customer_id"].to_numpy(),
                    "snapshot_date": snapshot["snapshot_date"].to_numpy(),
                    "segment": snapshot["segment"].to_numpy(),
                    "annual_contract_value": annual_contract_value,
                    "churn_probability": probabilities,
                    "revenue_at_risk": probabilities * annual_contract_value,
                }
            )
        )

        del snapshot, probabilities, annual_contract_value
        gc.collect()

    if not historical_results:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "snapshot_date",
                "segment",
                "annual_contract_value",
                "churn_probability",
                "revenue_at_risk",
            ]
        )

    historical = pd.concat(historical_results, ignore_index=True)
    del historical_results
    gc.collect()

    return historical


# ============================================================
# BUSINESS LAYER
# ============================================================

@st.cache_data(
    show_spinner="Calculating business risk..."
)
def prepare_business_layer(
    predictions
):

    result = predictions.copy()

    # --------------------------------------------------------
    # REVENUE AT RISK
    # --------------------------------------------------------

    result[
        "revenue_at_risk"
    ] = (
        result[
            "churn_probability"
        ]
        *
        result[
            "annual_contract_value"
        ].fillna(0)
    )

    # --------------------------------------------------------
    # RISK TIER
    # --------------------------------------------------------

    result[
        "risk_tier"
    ] = (
        result[
            "churn_probability"
        ]
        .apply(
            risk_tier
        )
    )

    # --------------------------------------------------------
    # EXPOSURE TIER
    # --------------------------------------------------------

    thresholds = {

        "moderate":
            result[
                "revenue_at_risk"
            ].quantile(0.50),

        "high":
            result[
                "revenue_at_risk"
            ].quantile(0.75),

        "critical":
            result[
                "revenue_at_risk"
            ].quantile(0.90),
    }

    result[
        "exposure_tier"
    ] = (
        result[
            "revenue_at_risk"
        ]
        .apply(
            lambda x:
            exposure_tier(
                x,
                thresholds
            )
        )
    )

    # --------------------------------------------------------
    # RENEWAL
    # --------------------------------------------------------

    result[
        "renewal_urgency"
    ] = (
        result[
            "days_to_renewal"
        ]
        .apply(
            renewal_urgency
        )
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    result[
        "health_signal"
    ] = result.apply(
        health_signal,
        axis=1
    )

    result[
        "usage_signal"
    ] = result.apply(
        usage_signal,
        axis=1
    )

    result[
        "support_signal"
    ] = result.apply(
        support_signal,
        axis=1
    )

    result[
        "recommended_action"
    ] = result.apply(
        recommended_action,
        axis=1
    )

    # --------------------------------------------------------
    # BUSINESS ASSUMPTIONS
    # --------------------------------------------------------

    success_rate = 0.30
    intervention_cost = 25_000

    result[
        "expected_save_value"
    ] = (
        result[
            "revenue_at_risk"
        ]
        * success_rate
    )

    result[
        "intervention_cost"
    ] = intervention_cost

    result[
        "net_expected_value"
    ] = (
        result[
            "expected_save_value"
        ]
        -
        result[
            "intervention_cost"
        ]
    )

    result[
        "expected_roi"
    ] = (
        result[
            "expected_save_value"
        ]
        /
        intervention_cost
    )

    return result


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

try:

    features = load_features()

    portfolio, latest_date = (
        prepare_portfolio(
            features
        )
    )

    predictions = (
        generate_predictions(
            portfolio
        )
    )

    predictions = (
        prepare_business_layer(
            predictions
        )
    )

except Exception as e:

    st.error(
        "Retain-AI could not initialize."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GLOBAL METRICS
# ============================================================

total_customers = len(
    predictions
)

total_acv = (
    predictions[
        "annual_contract_value"
    ]
    .fillna(0)
    .sum()
)

total_revenue_at_risk = (
    predictions[
        "revenue_at_risk"
    ].sum()
)

high_critical_count = len(
    predictions[
        predictions[
            "risk_tier"
        ].isin(
            [
                "High",
                "Critical"
            ]
        )
    ]
)

portfolio_exposure = (
    total_revenue_at_risk
    /
    total_acv
    if total_acv > 0
    else 0
)

average_probability = (
    predictions[
        "churn_probability"
    ].mean()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="padding:10px 2px 18px 2px;">

            <div class="brand-name">
                🎯 Retain-AI
            </div>

            <div class="brand-subtitle">
                Customer Retention<br>
                Intelligence Platform
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Workspace
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Customer Risk Explorer",
            "Customer 360",
            "Intervention Planner",
            "Model & Governance",
        ],
        label_visibility="collapsed",
    )

    st.html(
        """
        <div class="sidebar-label">
            Portfolio
        </div>
        """
    )

    st.html(
        f"""
        <div class="sidebar-stat">

            <div class="sidebar-stat-label">
                CUSTOMERS MONITORED
            </div>

            <div class="sidebar-stat-value">
                {total_customers:,}
            </div>

        </div>

        <div class="sidebar-stat">

            <div class="sidebar-stat-label">
                REVENUE AT RISK
            </div>

            <div class="sidebar-stat-value">
                {format_currency(
                    total_revenue_at_risk
                )}
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Model
        </div>

        <div class="sidebar-info">

            <b>Model</b><br>
            Calibrated XGBoost<br><br>

            <b>Prediction</b><br>
            90-day churn probability<br><br>

            <b>Business Priority</b><br>
            Revenue at Risk

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Snapshot
        </div>
        """
    )

    st.caption(
        latest_date.strftime(
            "%d %b %Y"
        )
    )

    st.html(
        """
        <div class="sidebar-info"
             style="margin-top:18px;">

            Predictions are generated using
            the production-calibrated XGBoost model.

            <br><br>

            Revenue-at-Risk and intervention
            economics are business decision metrics.

        </div>
        """
    )


# ============================================================
# APPLICATION HERO
# ============================================================

st.html(
    f"""
    <div class="hero">

        <div class="hero-title">
            Retain-AI
        </div>

        <div class="hero-subtitle">
            AI-powered customer retention decision platform
        </div>

        <div class="hero-date">
            Model inference snapshot:
            <b>{latest_date.strftime("%d %B %Y")}</b>
        </div>

    </div>
    """
)

st.html(
    f"""
    <div style="
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin:-8px 0 18px 0;
    ">
        <span class="status-chip status-chip-blue">● {total_customers:,} accounts monitored</span>
        <span class="status-chip status-chip-purple">● Calibrated XGBoost</span>
        <span class="status-chip status-chip-teal">● 90-day churn prediction</span>
        <span class="status-chip status-chip-blue">● Business priority: Revenue at Risk</span>
    </div>
    """
)


# ============================================================
# CUSTOMER 360 UI HELPERS
# ============================================================


def format_shap_value(value):
    """Format SHAP contribution."""

    value = float(value)

    return f"{value:+.3f}"


def shap_bar_width(shap_value, max_abs_shap):
    """
    Convert SHAP magnitude into a percentage width.
    """

    if max_abs_shap <= 0:
        return 5

    width = (
        abs(float(shap_value))
        / max_abs_shap
        * 100
    )

    return max(6, min(width, 100))


def render_shap_driver(
    driver,
    max_abs_shap,
    positive=True
):
    """
    Render one SHAP driver as a polished HTML card.
    """

    feature = html.escape(
        str(driver["feature"])
    )

    value = html.escape(
        str(driver["value"])
    )

    shap_value = float(
        driver["shap_value"]
    )

    shap_display = format_shap_value(
        shap_value
    )

    width = shap_bar_width(
        shap_value,
        max_abs_shap
    )

    if positive:

        accent = "#E54868"
        bar_class = "shap-bar-positive"
        symbol = "↑"

    else:

        accent = "#3B82F6"
        bar_class = "shap-bar-negative"
        symbol = "↓"

    return f"""
    <div class="shap-driver">

        <div class="shap-driver-header">

            <div class="shap-feature">
                {symbol}&nbsp;&nbsp;{feature}
            </div>

            <div class="shap-value">
                {shap_display}
            </div>

        </div>

        <div class="shap-driver-meta">
            <span>{value}</span>
            <span>SHAP contribution</span>
        </div>

        <div class="shap-track">

            <div
                class="{bar_class}"
                style="
                    width:{width}%;
                    background:{accent};
                "
            ></div>

        </div>

    </div>
    """


# ============================================================
# PAGE 1
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    st.write(
        "See where churn risk is concentrated, how much revenue is exposed, "
        "and where limited retention capacity should be deployed."
    )

    # --------------------------------------------------------
    # DEFAULT DECISION ENGINE VIEW
    # --------------------------------------------------------

    # The Executive Overview uses the same deterministic Decision Engine
    # used by the Intervention Planner. The 10% capacity is the default
    # operating point established during validation analysis.
    from decision_engine import build_decision_engine

    overview_decisions = build_decision_engine(
        predictions=predictions,
        intervention_success_rate=0.30,
        intervention_cost=25_000,
        intervention_capacity=0.10,
    )

    overview_selected = overview_decisions[
        overview_decisions[
            "selected_for_intervention"
        ]
    ].copy()

    overview_total_risk = overview_decisions[
        "revenue_at_risk"
    ].sum()

    overview_selected_risk = overview_selected[
        "revenue_at_risk"
    ].sum()

    overview_risk_coverage = (
        overview_selected_risk
        / overview_total_risk
        if overview_total_risk > 0
        else 0
    )

    overview_expected_save = overview_selected[
        "expected_save_value"
    ].sum()

    overview_cost = overview_selected[
        "intervention_cost"
    ].sum()

    overview_net_value = overview_selected[
        "net_expected_value"
    ].sum()

    overview_benefit_cost = (
        overview_expected_save
        / overview_cost
        if overview_cost > 0
        else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    customers_at_risk = len(
        overview_decisions[
            overview_decisions["risk_tier"] != "Low"
        ]
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:

        st.html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Customers at Risk</div>
                <div class="kpi-value">{customers_at_risk:,}</div>
                <div class="kpi-subtitle">Moderate, High or Critical</div>
            </div>
            """
        )

    with k2:

        st.html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Revenue at Risk</div>
                <div class="kpi-value">{format_currency(total_revenue_at_risk)}</div>
                <div class="kpi-subtitle">P(churn) × annual contract value</div>
            </div>
            """
        )

    with k3:

        st.html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">High / Critical</div>
                <div class="kpi-value">{high_critical_count:,}</div>
                <div class="kpi-subtitle">{high_critical_count / total_customers * 100:.1f}% of portfolio</div>
            </div>
            """
        )

    with k4:

        st.html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Selected for Action</div>
                <div class="kpi-value">{len(overview_selected):,}</div>
                <div class="kpi-subtitle">10% intervention capacity</div>
            </div>
            """
        )

    with k5:

        st.html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Net Value</div>
                <div class="kpi-value">{format_currency(overview_net_value)}</div>
                <div class="kpi-subtitle">Expected save less intervention cost</div>
            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # PORTFOLIO DECISION SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "Portfolio Decision Snapshot"
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Accounts selected",
            f"{len(overview_selected):,}",
            "10% capacity",
        )

    with d2:
        st.metric(
            "Revenue-at-Risk covered",
            format_currency(overview_selected_risk),
            f"{overview_risk_coverage * 100:.1f}% of portfolio risk",
        )

    with d3:
        st.metric(
            "Expected Save Value",
            format_currency(overview_expected_save),
            "30% assumed intervention success",
        )

    with d4:
        st.metric(
            "Expected Benefit / Cost",
            f"{overview_benefit_cost:.1f}×",
            "₹25K intervention cost / account",
        )

    st.caption(
        "Business assumptions: 30% intervention success rate and ₹25,000 intervention cost per account. "
        "These are explicit assumptions, not learned treatment effects."
    )

    st.write("")

    # --------------------------------------------------------
    # PORTFOLIO TRENDS
    # --------------------------------------------------------

    st.subheader(
        "Portfolio Risk Trend"
    )

    st.caption(
        "Historical model inference across the feature store. "
        "This view uses the production-calibrated XGBoost model; no churn labels are used."
    )

    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:

        historical_predictions = generate_historical_portfolio_predictions(
            features
        )

        monthly_risk = (
            historical_predictions
            .set_index("snapshot_date")
            ["churn_probability"]
            .resample("MS")
            .mean()
            .dropna()
            .to_frame("Average predicted churn risk")
        )

        st.write(
            "**Average predicted churn risk**"
        )

        risk_chart_data = monthly_risk.reset_index()

        risk_trend_chart = (
            alt.Chart(risk_chart_data)
            .mark_line(
                color="#4F6FF5",
                strokeWidth=3,
                point=alt.OverlayMarkDef(
                    filled=True,
                    color="#4F6FF5",
                    size=58,
                ),
            )
            .encode(
                x=alt.X(
                    "snapshot_date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b %Y",
                        labelColor="#526784",
                        labelFontSize=11,
                        labelAngle=0,
                        tickColor="#CBD7E6",
                        domainColor="#CBD7E6",
                    ),
                ),
                y=alt.Y(
                    "Average predicted churn risk:Q",
                    title="Average predicted churn risk",
                    axis=alt.Axis(
                        format=".1%",
                        labelColor="#526784",
                        titleColor="#405674",
                        titleFontSize=12,
                        labelFontSize=11,
                        gridColor="#E8EEF6",
                        domainColor="#CBD7E6",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "snapshot_date:T",
                        title="Month",
                        format="%B %Y",
                    ),
                    alt.Tooltip(
                        "Average predicted churn risk:Q",
                        title="Predicted churn risk",
                        format=".2%",
                    ),
                ],
            )
            .properties(height=285)
            .configure(
                background="#FFFFFF",
                padding={"left": 8, "right": 8, "top": 8, "bottom": 8},
            )
            .configure_view(
                fill="#FFFFFF",
                strokeOpacity=0,
            )
        )

        st.altair_chart(
            risk_trend_chart,
            use_container_width=True,
        )

    with trend_col2:

        monthly_exposure = (
            historical_predictions
            .set_index("snapshot_date")
            ["revenue_at_risk"]
            .resample("MS")
            .sum()
            .dropna()
            .to_frame("Revenue at Risk")
        )

        st.write(
            "**Revenue at Risk trend**"
        )

        exposure_chart_data = monthly_exposure.reset_index()
        exposure_chart_data["Revenue at Risk (₹ Cr)"] = (
            exposure_chart_data["Revenue at Risk"] / 1_00_00_000
        )

        exposure_trend_chart = (
            alt.Chart(exposure_chart_data)
            .mark_line(
                color="#7C3AED",
                strokeWidth=3,
                point=alt.OverlayMarkDef(
                    filled=True,
                    color="#7C3AED",
                    size=58,
                ),
            )
            .encode(
                x=alt.X(
                    "snapshot_date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b %Y",
                        labelColor="#526784",
                        labelFontSize=11,
                        labelAngle=0,
                        tickColor="#CBD7E6",
                        domainColor="#CBD7E6",
                    ),
                ),
                y=alt.Y(
                    "Revenue at Risk (₹ Cr):Q",
                    title="Revenue at Risk (₹ Cr)",
                    axis=alt.Axis(
                        format=".1f",
                        labelColor="#526784",
                        titleColor="#405674",
                        titleFontSize=12,
                        labelFontSize=11,
                        gridColor="#E8EEF6",
                        domainColor="#CBD7E6",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "snapshot_date:T",
                        title="Month",
                        format="%B %Y",
                    ),
                    alt.Tooltip(
                        "Revenue at Risk (₹ Cr):Q",
                        title="Revenue at Risk",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=285)
            .configure(
                background="#FFFFFF",
                padding={"left": 8, "right": 8, "top": 8, "bottom": 8},
            )
            .configure_view(
                fill="#FFFFFF",
                strokeOpacity=0,
            )
        )

        st.altair_chart(
            exposure_trend_chart,
            use_container_width=True,
        )

    trend_start = historical_predictions["snapshot_date"].min()
    trend_end = historical_predictions["snapshot_date"].max()
    latest_month = monthly_risk.iloc[-1, 0]
    first_month = monthly_risk.iloc[0, 0]
    risk_change = latest_month - first_month

    st.caption(
        f"Trend window: {trend_start.strftime('%d %b %Y')} → "
        f"{trend_end.strftime('%d %b %Y')}. "
        f"Average predicted churn risk changed by {risk_change * 100:+.2f} percentage points "
        "from the first to the latest monthly observation."
    )

    st.write("")

    # --------------------------------------------------------
    # RISK DISTRIBUTION + RISK CONCENTRATION
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Customer Risk Distribution"
        )

        st.caption(
            "Number of customers in each predicted 90-day churn risk tier."
        )

        risk_order = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        risk_counts = (
            predictions[
                "risk_tier"
            ]
            .value_counts()
            .reindex(
                risk_order,
                fill_value=0
            )
        )

        risk_chart_df = (
            risk_counts
            .rename_axis("Risk Tier")
            .reset_index(name="Customers")
        )

        risk_distribution_chart = (
            alt.Chart(risk_chart_df)
            .mark_bar(
                cornerRadiusTopRight=7,
                cornerRadiusBottomRight=7,
            )
            .encode(
                y=alt.Y(
                    "Risk Tier:N",
                    sort=risk_order,
                    title=None,
                    axis=alt.Axis(
                        labelColor="#405674",
                        labelFontSize=12,
                        labelFontWeight=600,
                        domainColor="#CBD7E6",
                        tickColor="#CBD7E6",
                    ),
                ),
                x=alt.X(
                    "Customers:Q",
                    title="Customers",
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#405674",
                        labelFontSize=11,
                        titleFontSize=12,
                        gridColor="#E8EEF6",
                        domainColor="#CBD7E6",
                    ),
                ),
                color=alt.Color(
                    "Risk Tier:N",
                    scale=alt.Scale(
                        domain=risk_order,
                        range=["#EF4444", "#F97316", "#F59E0B", "#22C55E"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Risk Tier:N", title="Risk"),
                    alt.Tooltip("Customers:Q", title="Customers", format=","),
                ],
            )
            .properties(height=285)
            .configure(
                background="#FFFFFF",
                padding={"left": 5, "right": 8, "top": 5, "bottom": 5},
            )
            .configure_view(fill="#FFFFFF", strokeOpacity=0)
        )

        st.altair_chart(
            risk_distribution_chart,
            use_container_width=True,
        )

    with right:

        st.subheader(
            "Revenue-at-Risk Concentration"
        )

        st.caption(
            "Share of total Revenue at Risk by customer segment."
        )

        segment_risk_chart = (
            predictions
            .groupby("segment")["revenue_at_risk"]
            .sum()
            .sort_values(ascending=False)
        )

        segment_risk_df = (
            segment_risk_chart
            .rename_axis("Segment")
            .reset_index(name="Revenue at Risk")
        )
        segment_risk_df["Revenue at Risk (₹ Cr)"] = (
            segment_risk_df["Revenue at Risk"] / 1_00_00_000
        )

        segment_concentration_chart = (
            alt.Chart(segment_risk_df)
            .mark_bar(
                cornerRadiusTopRight=7,
                cornerRadiusBottomRight=7,
            )
            .encode(
                y=alt.Y(
                    "Segment:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#405674",
                        labelFontSize=12,
                        labelFontWeight=600,
                        domainColor="#CBD7E6",
                        tickColor="#CBD7E6",
                    ),
                ),
                x=alt.X(
                    "Revenue at Risk (₹ Cr):Q",
                    title="Revenue at Risk (₹ Cr)",
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#405674",
                        labelFontSize=11,
                        titleFontSize=12,
                        gridColor="#E8EEF6",
                        domainColor="#CBD7E6",
                    ),
                ),
                color=alt.Color(
                    "Segment:N",
                    scale=alt.Scale(
                        domain=["Enterprise", "Mid-Market", "SMB"],
                        range=["#7C3AED", "#4F6FF5", "#14B8A6"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Segment:N", title="Segment"),
                    alt.Tooltip(
                        "Revenue at Risk (₹ Cr):Q",
                        title="Revenue at Risk",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=285)
            .configure(
                background="#FFFFFF",
                padding={"left": 5, "right": 8, "top": 5, "bottom": 5},
            )
            .configure_view(fill="#FFFFFF", strokeOpacity=0)
        )

        st.altair_chart(
            segment_concentration_chart,
            use_container_width=True,
        )

    st.write("")

    # --------------------------------------------------------
    # EXECUTIVE INSIGHT
    # --------------------------------------------------------

    segment_concentration = (
        overview_decisions
        .groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .reset_index()
    )

    segment_concentration["customer_share"] = (
        segment_concentration["customers"]
        / total_customers
    )

    segment_concentration["risk_share"] = (
        segment_concentration["revenue_at_risk"]
        / total_revenue_at_risk
        if total_revenue_at_risk > 0
        else 0
    )

    segment_concentration["risk_concentration_index"] = (
        segment_concentration["risk_share"]
        / segment_concentration["customer_share"].replace(0, np.nan)
    )

    concentration_leader = segment_concentration.loc[
        segment_concentration["risk_share"].idxmax()
    ]

    leader_name = concentration_leader["segment"]
    leader_customer_share = concentration_leader["customer_share"]
    leader_risk_share = concentration_leader["risk_share"]

    st.html(
        f"""
        <div class="insight-box">
            <div class="insight-label">Executive Insight</div>
            <div class="insight-text">
                <b>{leader_name} represents {leader_customer_share * 100:.1f}% of monitored customers
                but carries {leader_risk_share * 100:.1f}% of Revenue at Risk.</b>
                Retention capacity should therefore be allocated using economic exposure
                alongside predicted churn risk, rather than customer count alone.
            </div>
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # RISK CONCENTRATION TABLE
    # --------------------------------------------------------

    st.subheader(
        "Risk Concentration by Segment"
    )

    display_concentration = (
        segment_concentration[
            [
                "segment",
                "customers",
                "customer_share",
                "revenue_at_risk",
                "risk_share",
                "risk_concentration_index",
            ]
        ]
        .sort_values(
            "revenue_at_risk",
            ascending=False
        )
        .copy()
    )

    display_concentration["customer_share"] = (
        display_concentration["customer_share"]
        .map(lambda x: f"{x * 100:.1f}%")
    )

    display_concentration["revenue_at_risk"] = (
        display_concentration["revenue_at_risk"]
        .map(format_currency)
    )

    display_concentration["risk_share"] = (
        display_concentration["risk_share"]
        .map(lambda x: f"{x * 100:.1f}%")
    )

    display_concentration["risk_concentration_index"] = (
        display_concentration["risk_concentration_index"]
        .map(lambda x: f"{x:.2f}×")
    )

    display_concentration = display_concentration.rename(
        columns={
            "segment": "Segment",
            "customers": "Customers",
            "customer_share": "Customer Share",
            "revenue_at_risk": "Revenue at Risk",
            "risk_share": "Risk Share",
            "risk_concentration_index": "Risk Concentration",
        }
    )

    st.dataframe(
        display_concentration,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Risk Concentration = segment share of Revenue at Risk ÷ segment share of customers. "
        "Values above 1× indicate disproportionate economic exposure."
    )

    st.write("")

    # --------------------------------------------------------
    # EXECUTIVE AI
    # --------------------------------------------------------

    st.subheader("Executive AI")

    st.caption(
        "Qwen3:4B synthesizes the calibrated XGBoost portfolio risk, Decision Engine economics, "
        "risk concentration and historical model trends into a leadership brief. It does not make "
        "the churn prediction or change prioritisation."
    )

    if "executive_ai_insight" not in st.session_state:
        st.session_state["executive_ai_insight"] = None

    ai_col, info_col = st.columns([1, 3], gap="large")

    with ai_col:
        generate_executive_ai = st.button(
            "Generate Executive AI Brief",
            type="primary",
            use_container_width=True,
            key="generate_executive_ai_brief",
        )

    with info_col:
        st.info(
            "AI is grounded in the portfolio metrics shown above and uses the same "
            "production XGBoost + Decision Engine pipeline. Generation runs locally through Qwen3:4B."
        )

    if generate_executive_ai:
        try:
            from executive_insights import (
                prepare_executive_context,
                generate_executive_insight,
            )

            with st.spinner("Synthesising the executive portfolio brief..."):
                executive_context = prepare_executive_context(
                    overview_decisions=overview_decisions,
                    selected=overview_selected,
                    total_customers=total_customers,
                    total_acv=total_acv,
                    historical_predictions=historical_predictions,
                    monthly_risk=monthly_risk,
                    segment_concentration=segment_concentration,
                    high_critical_count=high_critical_count,
                    overview_expected_save=overview_expected_save,
                    overview_cost=overview_cost,
                    overview_net_value=overview_net_value,
                    overview_benefit_cost=overview_benefit_cost,
                    intervention_success_rate=0.30,
                    intervention_cost=25_000,
                    intervention_capacity=0.10,
                )

                st.session_state["executive_ai_insight"] = (
                    generate_executive_insight(executive_context)
                )

        except Exception as e:
            st.session_state["executive_ai_insight"] = None
            st.error(
                "Unable to generate the Executive AI brief. Make sure Ollama is running "
                "and Qwen3:4B is available."
            )
            st.exception(e)

    executive_ai = st.session_state.get("executive_ai_insight")

    if executive_ai:
        status_key = str(executive_ai["portfolio_status"]).lower()
        status_class = f"executive-ai-status-{status_key}"

        st.html(
            f"""
            <div class="executive-ai-card">
                <div class="executive-ai-kicker">Local AI • Qwen3:4B • Portfolio-grounded</div>
                <div class="executive-ai-title">Executive Retention Brief</div>
                <div class="executive-ai-summary">
                    {html.escape(str(executive_ai["executive_summary"]))}
                </div>
                <span class="executive-ai-status {status_class}">
                    Portfolio posture: {html.escape(str(executive_ai["portfolio_status"]))}
                </span>
            </div>
            """
        )

        st.write("")

        findings = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in executive_ai["key_findings"]
        )
        focus = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in executive_ai["priority_focus"]
        )
        actions = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in executive_ai["recommended_actions"]
        )
        watchouts = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in executive_ai["watchouts"]
        )

        ai1, ai2, ai3, ai4 = st.columns(4, gap="medium")

        for column, title, items in [
            (ai1, "What the portfolio is telling us", findings),
            (ai2, "Where leadership should focus", focus),
            (ai3, "Recommended actions", actions),
            (ai4, "Watchouts", watchouts),
        ]:
            with column:
                st.html(
                    f"""
                    <div class="executive-ai-section">
                        <div class="executive-ai-section-title">{html.escape(title)}</div>
                        <ul>{items}</ul>
                    </div>
                    """
                )

        st.html(
            """
            <div class="executive-ai-footer">
                AI-generated communication layer. Churn probability remains governed by the
                calibrated XGBoost model; financial prioritisation and intervention selection
                remain governed by the deterministic Decision Engine.
            </div>
            """
        )
    else:
        st.info(
            "Generate the Executive AI Brief to turn the portfolio analytics into a concise "
            "leadership-level retention narrative."
        )

    st.write("")

    # --------------------------------------------------------
    # TOP PRIORITY ACCOUNTS
    # --------------------------------------------------------

    st.subheader(
        "Top Retention Priorities"
    )

    priority_columns = [
        "priority_rank",
        "customer_id",
        "segment",
        "churn_probability",
        "annual_contract_value",
        "revenue_at_risk",
        "net_expected_value",
        "risk_tier",
        "days_to_renewal",
        "behaviour",
        "recommended_action",
    ]

    priority = (
        overview_selected
        .sort_values("priority_rank")
        [priority_columns]
        .head(10)
        .copy()
    )

    priority["churn_probability"] = (
        priority["churn_probability"]
        .map(format_probability)
    )

    for column in [
        "annual_contract_value",
        "revenue_at_risk",
        "net_expected_value",
    ]:
        priority[column] = (
            priority[column]
            .map(format_currency)
        )

    priority = priority.rename(
        columns={
            "priority_rank": "Priority",
            "customer_id": "Customer",
            "segment": "Segment",
            "churn_probability": "Churn Risk",
            "annual_contract_value": "ACV",
            "revenue_at_risk": "Revenue at Risk",
            "net_expected_value": "Net Expected Value",
            "risk_tier": "Risk",
            "days_to_renewal": "Renewal Days",
            "behaviour": "Behaviour",
            "recommended_action": "Recommended Action",
        }
    )

    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Priority is determined by expected business value after applying the Decision Engine's "
        "risk, exposure, intervention economics and capacity rules."
    )

    st.write("")

    # --------------------------------------------------------
    # HOW THE PORTFOLIO IS PRIORITISED
    # --------------------------------------------------------

    st.subheader(
        "How Retain-AI Prioritises the Portfolio"
    )

    st.markdown(
        """
        **1. Predict churn probability**  
        The calibrated XGBoost model estimates each customer's 90-day churn probability.

        **2. Quantify economic exposure**  
        `Revenue at Risk = P(churn) × Annual Contract Value`

        **3. Estimate intervention value**  
        `Expected Save Value = Revenue at Risk × assumed intervention success rate`

        **4. Account for intervention cost**  
        `Net Expected Value = Expected Save Value − intervention cost`

        **5. Allocate limited capacity**  
        The Decision Engine ranks customers by business value and selects the highest-priority
        accounts within the configured intervention capacity.
        """
    )


# ============================================================
# PAGE 2
# CUSTOMER RISK EXPLORER
# ============================================================

elif page == "Customer Risk Explorer":

    st.header(
        "Customer Risk Explorer"
    )

    st.write(
        "Filter and prioritise customers using predictive risk, "
        "economic exposure, behaviour and renewal context."
    )

    st.write("")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "Search Customer",
        placeholder="Enter customer ID, e.g. C04303",
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:

        segments = sorted(
            predictions[
                "segment"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_segments = st.multiselect(
            "Segment",
            segments,
            default=segments,
        )

    with f2:

        risk_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        selected_risk = st.multiselect(
            "Risk Tier",
            risk_options,
            default=risk_options,
        )

    with f3:

        exposure_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        selected_exposure = st.multiselect(
            "Exposure Tier",
            exposure_options,
            default=exposure_options,
        )

    with f4:

        behaviours = sorted(
            predictions[
                "behaviour"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_behaviour = st.multiselect(
            "Behaviour",
            behaviours,
            default=behaviours,
        )

    with f5:

        renewal_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
            "Very Low",
            "Unknown",
        ]

        selected_renewal = st.multiselect(
            "Renewal Urgency",
            renewal_options,
            default=renewal_options,
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    filtered = predictions[
        predictions[
            "segment"
        ].isin(
            selected_segments
        )
        &
        predictions[
            "risk_tier"
        ].isin(
            selected_risk
        )
        &
        predictions[
            "exposure_tier"
        ].isin(
            selected_exposure
        )
        &
        predictions[
            "behaviour"
        ].isin(
            selected_behaviour
        )
        &
        predictions[
            "renewal_urgency"
        ].isin(
            selected_renewal
        )
    ].copy()

    if search:

        search_value = (
            search
            .strip()
            .upper()
        )

        filtered = filtered[
            filtered[
                "customer_id"
            ]
            .astype(str)
            .str.upper()
            .str.contains(
                search_value,
                regex=False
            )
        ]

    # --------------------------------------------------------
    # FILTERED KPIs
    # --------------------------------------------------------

    filtered_risk = (
        filtered[
            "revenue_at_risk"
        ].sum()
    )

    filtered_average_risk = (
        filtered[
            "churn_probability"
        ].mean()
        if len(filtered)
        else 0
    )

    filtered_high = len(
        filtered[
            filtered[
                "risk_tier"
            ].isin(
                [
                    "High",
                    "Critical"
                ]
            )
        ]
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "Matching Customers",
            f"{len(filtered):,}"
        )

    with a2:

        st.metric(
            "Revenue at Risk",
            format_currency(
                filtered_risk
            )
        )

    with a3:

        st.metric(
            "Average Churn Risk",
            format_probability(
                filtered_average_risk
            )
        )

    with a4:

        st.metric(
            "High / Critical",
            f"{filtered_high:,}"
        )

    st.write("")

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "Priority Accounts"
    )

    display_columns = [
        "customer_id",
        "segment",
        "churn_probability",
        "risk_tier",
        "annual_contract_value",
        "revenue_at_risk",
        "days_to_renewal",
        "behaviour",
        "recommended_action",
    ]

    table = (
        filtered
        .sort_values(
            "revenue_at_risk",
            ascending=False
        )[
            display_columns
        ]
        .copy()
    )

    table[
        "churn_probability"
    ] = (
        table[
            "churn_probability"
        ]
        .map(
            format_probability
        )
    )

    table[
        "annual_contract_value"
    ] = (
        table[
            "annual_contract_value"
        ]
        .map(
            format_currency
        )
    )

    table[
        "revenue_at_risk"
    ] = (
        table[
            "revenue_at_risk"
        ]
        .map(
            format_currency
        )
    )

    table = (
        table.rename(
            columns={
                "customer_id":
                    "Customer",
                "segment":
                    "Segment",
                "churn_probability":
                    "Churn Risk",
                "risk_tier":
                    "Risk",
                "annual_contract_value":
                    "ACV",
                "revenue_at_risk":
                    "Revenue at Risk",
                "days_to_renewal":
                    "Renewal Days",
                "behaviour":
                    "Behaviour",
                "recommended_action":
                    "Recommended Action",
            }
        )
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "Customer": st.column_config.TextColumn("Customer", width="small"),
            "Segment": st.column_config.TextColumn("Segment", width="small"),
            "Churn Risk": st.column_config.TextColumn("Churn Risk", width="small"),
            "Risk": st.column_config.TextColumn("Risk", width="small"),
            "ACV": st.column_config.TextColumn("ACV", width="medium"),
            "Revenue at Risk": st.column_config.TextColumn("Revenue at Risk", width="medium"),
            "Renewal Days": st.column_config.NumberColumn("Renewal Days", format="%d"),
            "Behaviour": st.column_config.TextColumn("Behaviour", width="medium"),
            "Recommended Action": st.column_config.TextColumn("Recommended Action", width="large"),
        },
    )


# ============================================================
# PAGE 3
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.header(
        "Customer 360"
    )

    st.write(
        "A complete view of customer risk, commercial exposure, "
        "health, engagement and recommended intervention."
    )

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER SELECTOR
    # --------------------------------------------------------

    customer_ids = (
        predictions[
            "customer_id"
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )

    default_customer = (
        "C04303"
        if "C04303" in customer_ids
        else customer_ids[0]
    )

    selected_customer = st.selectbox(
        "Customer",
        customer_ids,
        index=customer_ids.index(
            default_customer
        ),
    )

    customer = predictions[
        predictions[
            "customer_id"
        ].astype(str)
        ==
        selected_customer
    ].iloc[0]

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER HEADER
    # --------------------------------------------------------

    badge_class = {
        "Critical":
            "risk-critical",
        "High":
            "risk-high",
        "Moderate":
            "risk-moderate",
        "Low":
            "risk-low",
    }.get(
        customer["risk_tier"],
        "risk-low"
    )

    st.html(
        f"""
        <div class="retain-card">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
            ">

                <div>

                    <div style="
                        color:#0F172A;
                        font-size:1.35rem;
                        font-weight:750;
                    ">
                        {selected_customer}
                    </div>

                    <div style="
                        color:#64748B;
                        font-size:0.80rem;
                        margin-top:5px;
                    ">
                        {customer["segment"]}
                        &nbsp; • &nbsp;
                        {customer["industry"]}
                        &nbsp; • &nbsp;
                        {customer["region"]}
                    </div>

                </div>

                <div class="risk-badge {badge_class}">
                    {customer["risk_tier"]}
                </div>

            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Churn Probability",
            format_probability(
                customer[
                    "churn_probability"
                ]
            ),
            help="Predicted probability of churn within the model horizon."
        )

    with c2:

        st.metric(
            "Revenue at Risk",
            format_currency(
                customer[
                    "revenue_at_risk"
                ]
            ),
            help="Churn probability multiplied by annual contract value."
        )

    with c3:

        st.metric(
            "Annual Contract Value",
            format_currency(
                customer[
                    "annual_contract_value"
                ]
            )
        )

    with c4:

        days = customer[
            "days_to_renewal"
        ]

        st.metric(
            "Days to Renewal",
            (
                f"{int(days):,}"
                if pd.notna(days)
                else "N/A"
            )
        )

    st.write("")

    # --------------------------------------------------------
    # THREE SIGNAL CARDS
    # --------------------------------------------------------

    s1, s2, s3 = st.columns(3)

    with s1:

        health_score = (
            f'{customer["health_score"]:.1f}'
            if pd.notna(
                customer["health_score"]
            )
            else "N/A"
        )

        health_trend = (
            f'{customer["health_trend"]:.3f}'
            if pd.notna(
                customer["health_trend"]
            )
            else "N/A"
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Customer Health
                </div>

                <div class="signal-subtitle">
                    Current health and trajectory
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Health Score
                    </div>

                    <div class="signal-value">
                        {health_score}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Health Trend
                    </div>

                    <div class="signal-value">
                        {health_trend}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Behaviour
                    </div>

                    <div class="signal-value">
                        {customer["behaviour"]}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["health_signal"]}
                </div>

            </div>
            """
        )

    with s2:

        active_users = (
            customer[
                "active_users"
            ]
        )

        sessions = (
            customer[
                "sessions"
            ]
        )

        products = (
            customer[
                "product_count"
            ]
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Product Engagement
                </div>

                <div class="signal-subtitle">
                    Usage and product adoption
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Active Users
                    </div>

                    <div class="signal-value">
                        {active_users:,.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Sessions
                    </div>

                    <div class="signal-value">
                        {sessions:,.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Products Adopted
                    </div>

                    <div class="signal-value">
                        {products:.0f}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["usage_signal"]}
                </div>

            </div>
            """
        )

    with s3:

        csat = customer[
            "avg_csat_12w"
        ]

        csat_value = (
            f"{csat:.2f}"
            if pd.notna(csat)
            else "No data"
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Customer Experience
                </div>

                <div class="signal-subtitle">
                    Support interaction profile
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Tickets — 12W
                    </div>

                    <div class="signal-value">
                        {customer["tickets_12w"]:.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Escalations — 12W
                    </div>

                    <div class="signal-value">
                        {customer["escalations_12w"]:.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Average CSAT — 12W
                    </div>

                    <div class="signal-value">
                        {csat_value}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["support_signal"]}
                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # RETENTION RECOMMENDATION
    # --------------------------------------------------------

    st.subheader(
        "Retention Recommendation"
    )

    st.success(
        customer[
            "recommended_action"
        ]
    )

    st.caption(
        "The recommendation combines customer-health, "
        "usage, support and renewal signals. "
        "Business priority is determined separately using Revenue at Risk."
    )

    st.write("")

    # --------------------------------------------------------
    # BUSINESS CONTEXT
    # --------------------------------------------------------

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(
            "Expected Save Value",
            format_currency(
                customer[
                    "expected_save_value"
                ]
            )
        )

    with b2:

        st.metric(
            "Intervention Cost",
            format_currency(
                customer[
                    "intervention_cost"
                ]
            )
        )

    with b3:

        st.metric(
            "Expected Benefit / Cost",
            f'{customer["expected_roi"]:.2f}×'
        )

    st.caption(
        "Economics assume a 30% intervention success rate "
        "and ₹25,000 intervention cost."
    )

    st.write("")

    # --------------------------------------------------------
    # AI RETENTION INSIGHT
    # --------------------------------------------------------

    st.subheader(
        "AI Retention Insight"
    )

    st.caption(
        "Qwen3:4B translates the calibrated XGBoost prediction, "
        "SHAP explanation and Decision Engine signals into an "
        "action-oriented retention brief. It does not make the churn prediction."
    )

    if "ai_insights" not in st.session_state:
        st.session_state["ai_insights"] = {}

    generate_ai_insight = st.button(
        "Generate AI Retention Insight",
        type="primary",
        use_container_width=False,
        key="generate_ai_retention_insight",
    )

    if generate_ai_insight:

        try:

            from decision_engine import build_decision_engine
            from explainability import explain_customer
            from llm_insights import (
                prepare_customer_context,
                generate_customer_insight,
            )

            with st.spinner(
                "Preparing model explanation and generating AI insight..."
            ):

                # ------------------------------------------------
                # SHAP explanation
                # ------------------------------------------------

                customer_row = (
                    portfolio[
                        portfolio["customer_id"].astype(str)
                        == str(selected_customer)
                    ]
                    .iloc[[0]]
                    .copy()
                )

                explanation_for_ai = explain_customer(
                    customer_row,
                    top_n=5
                )

                # ------------------------------------------------
                # Decision Engine — use the full portfolio so
                # priority/selection remain portfolio-relative.
                # ------------------------------------------------

                decisions_for_ai = build_decision_engine(
                    predictions=predictions,
                    intervention_success_rate=0.30,
                    intervention_cost=25_000,
                    intervention_capacity=0.10,
                )

                decision_customer = (
                    decisions_for_ai[
                        decisions_for_ai["customer_id"].astype(str)
                        == str(selected_customer)
                    ]
                    .iloc[0]
                )

                # ------------------------------------------------
                # Structured context → local LLM
                # ------------------------------------------------

                # Pass the three authoritative inputs separately.
                # The customer row carries profile/health/usage/support data,
                # the Decision Engine row carries churn risk and business
                # prioritisation, and SHAP carries model explanation.
                llm_context = prepare_customer_context(
                    customer,
                    decision_customer,
                    explanation_for_ai,
                )

                insight = generate_customer_insight(
                    llm_context
                )

                st.session_state["ai_insights"][
                    str(selected_customer)
                ] = insight

        except Exception as e:

            st.error(
                "Unable to generate the AI retention insight. "
                "Make sure Ollama is running and Qwen3:4B is available."
            )
            st.exception(e)

    # --------------------------------------------------------
    # DISPLAY CACHED INSIGHT
    # --------------------------------------------------------

    ai_insight = st.session_state["ai_insights"].get(
        str(selected_customer)
    )

    if ai_insight:

        st.html(
            f"""
            <div class="ai-insight-card">

                <div class="ai-insight-kicker">
                    Local AI • Qwen3:4B • Model-grounded
                </div>

                <div class="ai-insight-title">
                    Retention Brief for {html.escape(str(selected_customer))}
                </div>

                <div class="ai-insight-summary">
                    {html.escape(str(ai_insight["risk_summary"]))}
                </div>

            </div>
            """
        )

        st.write("")

        ai_priority, ai_action = st.columns(2, gap="large")

        with ai_priority:

            priority_class = {
                "Critical": "risk-critical",
                "High": "risk-high",
                "Moderate": "risk-moderate",
                "Low": "risk-low",
            }.get(
                str(ai_insight["priority"]),
                "risk-low",
            )

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">AI Priority</div>
                    <span class="risk-badge {priority_class}">
                        {html.escape(str(ai_insight["priority"]))}
                    </span>
                </div>
                """
            )

        with ai_action:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Recommended Action</div>
                    <div class="retain-card-title">
                        {html.escape(str(ai_insight["recommended_action"]))}
                    </div>
                </div>
                """
            )

        st.write("")

        ai1, ai2 = st.columns(2, gap="large")

        driver_items = "".join(
            f"<li>{html.escape(str(driver))}</li>"
            for driver in ai_insight["key_drivers"]
        )

        step_items = "".join(
            f"<li>{html.escape(str(step))}</li>"
            for step in ai_insight["next_steps"]
        )

        with ai1:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Key Risk Drivers</div>
                    <ul style="
                        margin:8px 0 0 18px;
                        padding:0;
                        color:#334155;
                        line-height:1.7;
                        font-size:0.86rem;
                    ">
                        {driver_items}
                    </ul>
                </div>
                """
            )

        with ai2:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Next Steps</div>
                    <ol style="
                        margin:8px 0 0 18px;
                        padding:0;
                        color:#334155;
                        line-height:1.7;
                        font-size:0.86rem;
                    ">
                        {step_items}
                    </ol>
                </div>
                """
            )

        st.write("")

        st.html(
            f"""
            <div class="retain-card">
                <div class="ai-insight-label">Why This Matters</div>
                <div class="retain-card-subtitle" style="
                    font-size:0.86rem;
                    color:#334155;
                    line-height:1.65;
                ">
                    {html.escape(str(ai_insight["reasoning"]))}
                </div>
            </div>
            """
        )

        st.caption(
            "AI-generated communication layer. Churn probability, SHAP explanations "
            "and retention recommendations remain governed by the production model "
            "and Decision Engine."
        )

    else:

        st.info(
            "Select a customer and click **Generate AI Retention Insight** "
            "to create a grounded retention brief."
        )

    st.write("")

    # --------------------------------------------------------
    # SHAP AREA
    # --------------------------------------------------------

    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    st.header("Why is this customer at risk?")

    st.caption(
        "SHAP explains which customer attributes are "
        "pushing the model prediction higher or lower."
    )

    st.write("")
    
    try:
    
        from explainability import explain_customer
    
        # --------------------------------------------------------
        # Customer row
        # --------------------------------------------------------
    
        customer_row = (
            portfolio[
                portfolio["customer_id"].astype(str)
                == str(selected_customer)
            ]
            .iloc[[0]]
            .copy()
        )
    
        # --------------------------------------------------------
        # SHAP
        # --------------------------------------------------------
    
        explanation = explain_customer(
            customer_row,
            top_n=5
        )
    
        positive_drivers = (
            explanation["positive_drivers"]
        )
    
        negative_drivers = (
            explanation["negative_drivers"]
        )
    
        # --------------------------------------------------------
        # Determine common scale
        # --------------------------------------------------------
    
        all_drivers = (
            positive_drivers
            + negative_drivers
        )
    
        max_abs_shap = max(
            [
                abs(
                    float(x["shap_value"])
                )
                for x in all_drivers
            ],
            default=1
        )
    
        # --------------------------------------------------------
        # Render cards
        # --------------------------------------------------------
    
        left, right = st.columns(
            2,
            gap="large"
        )
    
        # ========================================================
        # INCREASING RISK
        # ========================================================
    
        with left:
        
            positive_html = ""
    
            for driver in positive_drivers:
            
                positive_html += render_shap_driver(
                    driver,
                    max_abs_shap,
                    positive=True
                )
    
            if not positive_html:
            
                positive_html = """
                <div style="
                    color:#8a94a6;
                    font-size:14px;
                    padding:20px 0;
                ">
                    No significant risk-increasing
                    drivers identified.
                </div>
                """
    
            st.html(
                f"""
                <div class="shap-column">
    
                    <div class="shap-column-title">
                        <span class="shap-icon-risk">▲</span>
                        Increasing churn risk
                    </div>
    
                    {positive_html}
    
                    <div class="shap-explanation-note">
                        Positive SHAP values increase the
                        model's predicted churn risk.
                    </div>
    
                </div>
                """
            )
    
        # ========================================================
        # REDUCING RISK
        # ========================================================
    
        with right:
        
            negative_html = ""
    
            for driver in negative_drivers:
            
                negative_html += render_shap_driver(
                    driver,
                    max_abs_shap,
                    positive=False
                )
    
            if not negative_html:
            
                negative_html = """
                <div style="
                    color:#8a94a6;
                    font-size:14px;
                    padding:20px 0;
                ">
                    No significant risk-reducing
                    drivers identified.
                </div>
                """
    
            st.html(
                f"""
                <div class="shap-column">
    
                    <div class="shap-column-title">
                        <span class="shap-icon-safe">▼</span>
                        Reducing churn risk
                    </div>
    
                    {negative_html}
    
                    <div class="shap-explanation-note">
                        Negative SHAP values reduce the
                        model's predicted churn risk.
                    </div>
    
                </div>
                """
            )
    
    
    except Exception as e:
    
        st.error(
            "Unable to generate customer explanation."
        )
    
        st.exception(e)

# ============================================================
# PAGE 4
# INTERVENTION PLANNER
# ============================================================

elif page == "Intervention Planner":

    st.header(
        "Intervention Planner"
    )

    st.write(
        "Translate predictive churn risk into an actionable retention plan "
        "under a finite Customer Success capacity."
    )

    st.write("")

    # --------------------------------------------------------
    # CAPACITY CONTROL
    # --------------------------------------------------------

    st.subheader(
        "Intervention Capacity"
    )

    st.caption(
        "Choose how much of the monitored portfolio your retention team can actively engage. "
        "Retain-AI then selects the highest-value accounts using the Decision Engine."
    )

    capacity_options = [
        "1%",
        "2%",
        "5%",
        "10%",
        "15%",
        "20%",
    ]

    capacity_map = {
        "1%": 0.01,
        "2%": 0.02,
        "5%": 0.05,
        "10%": 0.10,
        "15%": 0.15,
        "20%": 0.20,
    }

    selected_capacity_label = st.select_slider(
        "Capacity",
        options=capacity_options,
        value="10%",
        label_visibility="collapsed",
    )

    selected_capacity = capacity_map[
        selected_capacity_label
    ]

    st.write("")

    # --------------------------------------------------------
    # DECISION ENGINE
    # --------------------------------------------------------

    from decision_engine import build_decision_engine

    INTERVENTION_SUCCESS_RATE = 0.30
    INTERVENTION_COST = 25_000

    decisions = build_decision_engine(
        predictions=predictions,
        intervention_success_rate=INTERVENTION_SUCCESS_RATE,
        intervention_cost=INTERVENTION_COST,
        intervention_capacity=selected_capacity,
    )

    selected = decisions[
        decisions["selected_for_intervention"]
    ].copy()

    total_portfolio_risk = decisions[
        "revenue_at_risk"
    ].sum()

    selected_risk = selected[
        "revenue_at_risk"
    ].sum()

    risk_coverage = (
        selected_risk / total_portfolio_risk
        if total_portfolio_risk > 0
        else 0
    )

    expected_save = selected[
        "expected_save_value"
    ].sum()

    intervention_cost = selected[
        "intervention_cost"
    ].sum()

    net_expected_value = selected[
        "net_expected_value"
    ].sum()

    expected_benefit_cost = (
        expected_save / intervention_cost
        if intervention_cost > 0
        else 0
    )

    # --------------------------------------------------------
    # SELECTED SCENARIO KPIs
    # --------------------------------------------------------

    st.subheader(
        "Selected Allocation"
    )

    p1, p2, p3 = st.columns(3)

    first_row = [
        (
            "Customers Targeted",
            f"{len(selected):,}",
            f"{selected_capacity_label} of portfolio",
        ),
        (
            "Revenue-at-Risk Covered",
            format_currency(selected_risk),
            f"{risk_coverage * 100:.1f}% of portfolio risk",
        ),
        (
            "Expected Save Value",
            format_currency(expected_save),
            "30% assumed intervention success",
        ),
    ]

    for column, values in zip(
        [p1, p2, p3],
        first_row,
    ):
        with column:
            st.html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{values[0]}</div>
                    <div class="kpi-value">{values[1]}</div>
                    <div class="kpi-subtitle">{values[2]}</div>
                </div>
                """
            )

    st.write("")

    p4, p5, p6 = st.columns(3)

    second_row = [
        (
            "Intervention Cost",
            format_currency(intervention_cost),
            "₹25,000 per targeted account",
        ),
        (
            "Net Expected Value",
            format_currency(net_expected_value),
            "Expected save − intervention cost",
        ),
        (
            "Expected Benefit / Cost",
            f"{expected_benefit_cost:.2f}×",
            "Expected save ÷ intervention cost",
        ),
    ]

    for column, values in zip(
        [p4, p5, p6],
        second_row,
    ):
        with column:
            st.html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{values[0]}</div>
                    <div class="kpi-value">{values[1]}</div>
                    <div class="kpi-subtitle">{values[2]}</div>
                </div>
                """
            )

    st.write("")

    st.info(
        "Expected economics use a 30% intervention success assumption and "
        "₹25,000 intervention cost per account. These are explicit business "
        "assumptions, not model-learned treatment effects."
    )

    st.write("")

    # --------------------------------------------------------
    # CAPACITY SCENARIO ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "Capacity Scenario Analysis"
    )

    st.caption(
        "Compare how changing retention capacity changes the number of accounts "
        "targeted, risk coverage and expected economic value."
    )

    scenario_rows = []

    for capacity_label in capacity_options:

        capacity = capacity_map[capacity_label]

        scenario_decisions = build_decision_engine(
            predictions=predictions,
            intervention_success_rate=INTERVENTION_SUCCESS_RATE,
            intervention_cost=INTERVENTION_COST,
            intervention_capacity=capacity,
        )

        scenario_selected = scenario_decisions[
            scenario_decisions["selected_for_intervention"]
        ]

        scenario_risk = scenario_selected[
            "revenue_at_risk"
        ].sum()

        scenario_expected_save = scenario_selected[
            "expected_save_value"
        ].sum()

        scenario_cost = scenario_selected[
            "intervention_cost"
        ].sum()

        scenario_net_value = scenario_selected[
            "net_expected_value"
        ].sum()

        scenario_benefit_cost = (
            scenario_expected_save / scenario_cost
            if scenario_cost > 0
            else 0
        )

        scenario_coverage = (
            scenario_risk / total_portfolio_risk
            if total_portfolio_risk > 0
            else 0
        )

        scenario_rows.append(
            {
                "Capacity": capacity_label,
                "Customers Targeted": len(scenario_selected),
                "Revenue-at-Risk Covered": scenario_risk,
                "Risk Coverage": scenario_coverage,
                "Expected Save": scenario_expected_save,
                "Intervention Cost": scenario_cost,
                "Net Expected Value": scenario_net_value,
                "Benefit / Cost": scenario_benefit_cost,
            }
        )

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    scenario_display = scenario_df.copy()
    scenario_display["Revenue-at-Risk Covered"] = scenario_display[
        "Revenue-at-Risk Covered"
    ].map(format_currency)
    scenario_display["Risk Coverage"] = scenario_display[
        "Risk Coverage"
    ].map(lambda x: f"{x * 100:.1f}%")
    scenario_display["Expected Save"] = scenario_display[
        "Expected Save"
    ].map(format_currency)
    scenario_display["Intervention Cost"] = scenario_display[
        "Intervention Cost"
    ].map(format_currency)
    scenario_display["Net Expected Value"] = scenario_display[
        "Net Expected Value"
    ].map(format_currency)
    scenario_display["Benefit / Cost"] = scenario_display[
        "Benefit / Cost"
    ].map(lambda x: f"{x:.2f}×")

    styled_scenarios = (
        scenario_display.style
        .set_properties(
            **{
                "background-color": "#FFFFFF",
                "color": "#29405F",
                "border-color": "#DCE7F3",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#EEF4FF"),
                        ("color", "#294B78"),
                        ("font-weight", "700"),
                    ],
                }
            ]
        )
    )

    st.dataframe(
        styled_scenarios,
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    chart_left, chart_right = st.columns(2)

    chart_data = scenario_df.set_index(
        "Capacity"
    ).copy()

    chart_data["Risk Coverage (%)"] = (
        chart_data["Risk Coverage"] * 100
    )

    chart_data["Net Expected Value (₹ Cr)"] = (
        chart_data["Net Expected Value"] / 1_00_00_000
    )

    with chart_left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Risk coverage by intervention capacity**")
        coverage_chart = (
            alt.Chart(
                chart_data.reset_index()
            )
            .mark_line(
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=80,
                    color="#4F6FF5",
                ),
                strokeWidth=3,
                color="#4F6FF5",
            )
            .encode(
                x=alt.X(
                    "Capacity:N",
                    sort=capacity_options,
                    title=None,
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                y=alt.Y(
                    "Risk Coverage (%):Q",
                    title="Coverage (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Capacity:N", title="Capacity"),
                    alt.Tooltip(
                        "Risk Coverage (%):Q",
                        title="Risk Coverage",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=285)
            .configure(background="#FFFFFF", padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
            .configure_view(fill="#FFFFFF", strokeOpacity=0)
            .configure_axis(gridColor="#E8EEF6", domainColor="#CBD7E6", tickColor="#CBD7E6", labelColor="#526784", titleColor="#405674", labelFontSize=11, titleFontSize=12)
        )
        st.altair_chart(coverage_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Net expected value by intervention capacity (₹ Cr)**")
        value_chart = (
            alt.Chart(
                chart_data.reset_index()
            )
            .mark_line(
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=80,
                    color="#7C3AED",
                ),
                strokeWidth=3,
                color="#7C3AED",
            )
            .encode(
                x=alt.X(
                    "Capacity:N",
                    sort=capacity_options,
                    title=None,
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                y=alt.Y(
                    "Net Expected Value (₹ Cr):Q",
                    title="Net Expected Value (₹ Cr)",
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Capacity:N", title="Capacity"),
                    alt.Tooltip(
                        "Net Expected Value (₹ Cr):Q",
                        title="Net Expected Value",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=285)
            .configure(background="#FFFFFF", padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
            .configure_view(fill="#FFFFFF", strokeOpacity=0)
            .configure_axis(gridColor="#E8EEF6", domainColor="#CBD7E6", tickColor="#CBD7E6", labelColor="#526784", titleColor="#405674", labelFontSize=11, titleFontSize=12)
        )
        st.altair_chart(value_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------
    # RECOMMENDED ALLOCATION
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="insight-box">
            <div class="insight-label">Recommended Allocation</div>
            <div class="insight-text">
                At <b>{selected_capacity_label}</b> intervention capacity,
                Retain-AI recommends engaging <b>{len(selected):,} accounts</b>.
                The selected group covers <b>{risk_coverage * 100:.1f}%</b> of
                portfolio Revenue at Risk, representing <b>{format_currency(expected_save)}</b>
                of expected save value after applying the stated success assumption.
                Selection is driven by expected business value, not churn probability alone.
            </div>
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # SELECTED PORTFOLIO COMPOSITION
    # --------------------------------------------------------

    st.subheader(
        "Selected Portfolio Composition"
    )

    composition_left, composition_right = st.columns(2)

    with composition_left:

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Revenue-at-Risk selected by customer segment (₹ Cr)**")

        segment_selected = (
            selected
            .groupby("segment")["revenue_at_risk"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        segment_selected["Revenue at Risk (₹ Cr)"] = (
            segment_selected["revenue_at_risk"] / 1_00_00_000
        )

        segment_chart = (
            alt.Chart(segment_selected)
            .mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6,
                color="#4F6FF5",
            )
            .encode(
                y=alt.Y(
                    "segment:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#405674",
                        labelFontSize=12,
                    ),
                ),
                x=alt.X(
                    "Revenue at Risk (₹ Cr):Q",
                    title="₹ Cr",
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("segment:N", title="Segment"),
                    alt.Tooltip(
                        "Revenue at Risk (₹ Cr):Q",
                        title="Revenue at Risk",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=285)
            .configure(background="#FFFFFF", padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
            .configure_view(fill="#FFFFFF", strokeOpacity=0)
            .configure_axis(gridColor="#E8EEF6", domainColor="#CBD7E6", tickColor="#CBD7E6", labelColor="#526784", titleColor="#405674", labelFontSize=11, titleFontSize=12)
        )
        st.altair_chart(segment_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with composition_right:

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Targeted accounts by primary intervention**")

        action_mix = (
            selected["recommended_action"]
            .fillna("Unspecified")
            .astype(str)
            .str.split(" • ")
            .str[0]
            .value_counts()
            .rename_axis("Intervention")
            .reset_index(name="Customers")
        )

        action_labels = {
            "Account Health Intervention": "Account Health",
            "Customer Success Monitoring": "Customer Success",
            "Product Adoption Intervention": "Product Adoption",
            "Early Engagement Warning": "Early Engagement",
            "Technical / Customer Experience Escalation": "CX / Technical",
            "Customer Experience Review": "CX Review",
            "Renewal Engagement": "Renewal",
            "Proactive Renewal Engagement": "Proactive Renewal",
            "Proactive Monitoring": "Proactive Monitoring",
        }
        action_mix["Intervention"] = action_mix["Intervention"].map(
            lambda x: action_labels.get(x, x)
        )

        # Keep the chart readable when several intervention types exist.
        if len(action_mix) > 8:
            action_mix = action_mix.sort_values(
                "Customers",
                ascending=False,
            )
            other_count = action_mix.iloc[8:]["Customers"].sum()
            action_mix = pd.concat(
                [
                    action_mix.iloc[:8],
                    pd.DataFrame(
                        [{"Intervention": "Other", "Customers": other_count}]
                    ),
                ],
                ignore_index=True,
            )

        action_mix = action_mix.sort_values(
            "Customers",
            ascending=False,
        )

        action_chart = (
            alt.Chart(action_mix)
            .mark_bar(
                cornerRadiusTopRight=7,
                cornerRadiusBottomRight=7,
            )
            .encode(
                y=alt.Y(
                    "Intervention:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#405674",
                        labelFontSize=11,
                        labelLimit=170,
                    ),
                ),
                x=alt.X(
                    "Customers:Q",
                    title="Accounts",
                    axis=alt.Axis(
                        labelColor="#526784",
                        titleColor="#526784",
                    ),
                ),
                color=alt.Color(
                    "Intervention:N",
                    scale=alt.Scale(
                        range=[
                            "#4F6FF5",
                            "#14B8A6",
                            "#7C3AED",
                            "#F59E0B",
                            "#EC4899",
                            "#0EA5E9",
                            "#F97316",
                            "#64748B",
                            "#94A3B8",
                        ]
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Intervention:N", title="Intervention"),
                    alt.Tooltip("Customers:Q", title="Accounts", format=","),
                ],
                order=alt.Order(
                    "Customers:Q",
                    sort="descending",
                ),
            )
            .properties(height=max(285, min(430, 36 * len(action_mix) + 55)))
            .configure(
                background="#FFFFFF",
                padding={"left": 8, "right": 8, "top": 8, "bottom": 8},
            )
            .configure_view(strokeOpacity=0)
            .configure_axis(gridColor="#E6EDF6")
        )
        st.altair_chart(action_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------
    # PRIORITY LIST
    # --------------------------------------------------------

    st.subheader(
        "Intervention Priority List"
    )

    st.caption(
        "Accounts selected by the Decision Engine, ranked by expected business value."
    )

    priority_columns = [
        "priority_rank",
        "customer_id",
        "segment",
        "churn_probability",
        "annual_contract_value",
        "revenue_at_risk",
        "expected_save_value",
        "net_expected_value",
        "risk_tier",
        "days_to_renewal",
        "behaviour",
        "recommended_action",
    ]

    priority = (
        selected
        .sort_values("priority_rank")
        [priority_columns]
        .head(100)
        .copy()
    )

    priority["churn_probability"] = priority[
        "churn_probability"
    ].map(format_probability)

    for column in [
        "annual_contract_value",
        "revenue_at_risk",
        "expected_save_value",
        "net_expected_value",
    ]:
        priority[column] = priority[column].map(
            format_currency
        )

    priority["recommended_action"] = (
        priority["recommended_action"]
        .astype(str)
        .str.replace(
            "Account Health Review",
            "Account Health",
            regex=False,
        )
        .str.replace(
            "CX / Technical Escalation",
            "CX / Technical",
            regex=False,
        )
        .str.replace(
            "Proactive Customer Monitoring",
            "Proactive Monitoring",
            regex=False,
        )
    )

    priority = priority.rename(
        columns={
            "priority_rank": "Priority",
            "customer_id": "Customer",
            "segment": "Segment",
            "churn_probability": "Churn Risk",
            "annual_contract_value": "ACV",
            "revenue_at_risk": "Revenue at Risk",
            "expected_save_value": "Expected Save",
            "net_expected_value": "Net Expected Value",
            "risk_tier": "Risk",
            "days_to_renewal": "Renewal Days",
            "behaviour": "Behaviour",
            "recommended_action": "Recommended Action",
        }
    )

    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.write("")

    # --------------------------------------------------------
    # DOWNLOADABLE ACTION LIST
    # --------------------------------------------------------

    st.subheader(
        "Export Intervention List"
    )

    export_columns = [
        "priority_rank",
        "customer_id",
        "segment",
        "churn_probability",
        "annual_contract_value",
        "revenue_at_risk",
        "expected_save_value",
        "intervention_cost",
        "net_expected_value",
        "risk_tier",
        "exposure_tier",
        "renewal_urgency",
        "days_to_renewal",
        "behaviour",
        "health_signal",
        "usage_signal",
        "support_signal",
        "recommended_action",
    ]

    export_df = (
        selected
        .sort_values("priority_rank")
        [export_columns]
        .copy()
    )

    st.caption(
        f"Export the {len(export_df):,} accounts selected at {selected_capacity_label} capacity, "
        "including risk, financial exposure, renewal urgency and recommended action."
    )

    st.download_button(
        label="⬇  Download intervention plan (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"retain_ai_intervention_plan_{selected_capacity_label.replace('%', 'pct')}.csv",
        mime="text/csv",
        type="primary",
    )

    st.write("")

    # --------------------------------------------------------
    # PRIORITISATION LOGIC
    # --------------------------------------------------------

    st.subheader(
        "How Retain-AI Prioritises Customers"
    )

    st.markdown(
        """
        **1. Predict churn probability**

        The calibrated XGBoost model estimates each customer's 90-day churn probability.

        **2. Quantify economic exposure**

        `Revenue at Risk = P(churn) × Annual Contract Value`

        **3. Estimate intervention value**

        `Expected Save Value = Revenue at Risk × assumed intervention success rate`

        **4. Account for intervention cost**

        `Net Expected Value = Expected Save Value − intervention cost`

        **5. Allocate finite capacity**

        The Decision Engine ranks economically viable accounts and selects the highest-priority
        customers within the configured intervention capacity.
        """
    )

    st.caption(
        "Priority is portfolio-relative. High churn probability does not automatically imply the highest business priority; economic exposure is considered alongside risk."
    )



# ============================================================
# PAGE 5
# MODEL & GOVERNANCE
# ============================================================

elif page == "Model & Governance":

    st.header("Model & Governance")

    st.write(
        "A transparent view of the production model, business assumptions, "
        "decision architecture and deployment controls behind Retain-AI."
    )

    st.write("")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Production Model Performance")

    st.caption(
        "Frozen evaluation results from the untouched test period. "
        "These metrics are reported for governance and monitoring; the test set "
        "is not used to tune the production operating point."
    )

    g1, g2, g3, g4 = st.columns(4)

    governance_metrics = [
        ("ROC-AUC", "0.8082", "Ranking discrimination"),
        ("PR-AUC", "0.2245", "Precision-recall performance"),
        ("Brier Score", "0.0468", "Probability calibration"),
        ("Prediction Horizon", "90 days", "Churn label horizon"),
    ]

    for column, (label, value, note) in zip(
        [g1, g2, g3, g4],
        governance_metrics,
    ):
        with column:
            st.html(
                f"""
                <div class="governance-card">
                    <div class="governance-kicker">{html.escape(label)}</div>
                    <div class="governance-value">{html.escape(value)}</div>
                    <div class="governance-note">{html.escape(note)}</div>
                </div>
                """
            )

    st.write("")

    st.subheader("Operating Point")

    o1, o2, o3, o4 = st.columns(4)

    operating_metrics = [
        ("Intervention Capacity", "10%", "Default portfolio capacity"),
        ("Precision", "23.88%", "Customers targeted who churn"),
        ("Recall", "43.48%", "Churners captured"),
        ("Lift", "4.35×", "Versus portfolio baseline"),
    ]

    for column, (label, value, note) in zip(
        [o1, o2, o3, o4],
        operating_metrics,
    ):
        with column:
            st.html(
                f"""
                <div class="governance-card">
                    <div class="governance-kicker">{html.escape(label)}</div>
                    <div class="governance-value">{html.escape(value)}</div>
                    <div class="governance-note">{html.escape(note)}</div>
                </div>
                """
            )

    st.caption(
        "The 10% operating point is a configurable business capacity, not a universal model threshold. "
        "The Intervention Planner allows 1%, 2%, 5%, 10%, 15% and 20% capacity scenarios."
    )

    st.write("")

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Champion Model Selection")

    comparison_df = pd.DataFrame(
        [
            ["Calibrated Logistic Regression", 0.7943, 0.2460, 0.0464, "Baseline"],
            ["Calibrated Random Forest", 0.7932, 0.2352, 0.0469, "Benchmark"],
            ["Original XGBoost", 0.7699, 0.1942, 0.0808, "Rejected"],
            ["Regularized + Calibrated XGBoost", 0.8016, 0.2466, 0.0467, "Champion"],
        ],
        columns=[
            "Model",
            "Validation ROC-AUC",
            "Validation PR-AUC",
            "Validation Brier",
            "Decision",
        ],
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="large"),
            "Validation ROC-AUC": st.column_config.NumberColumn(
                "ROC-AUC", format="%.4f"
            ),
            "Validation PR-AUC": st.column_config.NumberColumn(
                "PR-AUC", format="%.4f"
            ),
            "Validation Brier": st.column_config.NumberColumn(
                "Brier", format="%.4f"
            ),
            "Decision": st.column_config.TextColumn("Decision", width="small"),
        },
    )

    st.caption(
        "The calibrated XGBoost model was selected because it delivered the strongest validation ROC-AUC and PR-AUC "
        "while maintaining probability calibration comparable to the logistic baseline."
    )

    st.write("")

    # --------------------------------------------------------
    # BUSINESS NARRATIVE
    # --------------------------------------------------------

    st.subheader("What Retain-AI Does")

    st.html(
        """
        <div class="governance-panel">
            <div class="governance-panel-title">
                From churn prediction to retention decisions
            </div>
            <div class="governance-panel-text">
                <b>Retain-AI identifies customers likely to churn, quantifies the revenue exposed,
                explains the drivers behind risk, and directs limited retention resources toward
                the highest expected-value opportunities.</b>
                <br><br>
                The platform is intentionally designed as a decision system rather than a standalone
                prediction model. Probability comes from the calibrated XGBoost model; explanations
                come from SHAP; financial prioritisation and recommended actions come from the
                deterministic Decision Engine; and Qwen3:4B converts those grounded outputs into
                executive and customer-level communication.
            </div>
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader("AI Decision Architecture")

    st.html(
        """
        <div class="governance-panel">
            <div class="architecture-flow">

                <div class="architecture-node">
                    <strong>Feature Store</strong>
                    <span>Validated customer, usage, support and contract features</span>
                </div>

                <div class="architecture-arrow">→</div>

                <div class="architecture-node">
                    <strong>Calibrated XGBoost</strong>
                    <span>90-day churn probability</span>
                </div>

                <div class="architecture-arrow">→</div>

                <div class="architecture-node">
                    <strong>SHAP</strong>
                    <span>Customer and portfolio risk explanation</span>
                </div>

                <div class="architecture-arrow">→</div>

                <div class="architecture-node">
                    <strong>Decision Engine</strong>
                    <span>Revenue at Risk, EV and intervention priority</span>
                </div>

                <div class="architecture-arrow">→</div>

                <div class="architecture-node">
                    <strong>Qwen3:4B</strong>
                    <span>Grounded AI communication layer</span>
                </div>

                <div class="architecture-arrow">→</div>

                <div class="architecture-node">
                    <strong>Streamlit</strong>
                    <span>Executive and customer decision interface</span>
                </div>

            </div>
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # BUSINESS ASSUMPTIONS
    # --------------------------------------------------------

    st.subheader("Business Assumptions")

    assumptions_df = pd.DataFrame(
        [
            ["Intervention success rate", "30%", "Scenario assumption used to estimate expected save value"],
            ["Intervention cost", "₹25,000 / account", "Assumed cost of one retention intervention"],
            ["Default intervention capacity", "10%", "Default operating capacity; configurable in planner"],
            ["Revenue at Risk", "P(churn) × ACV", "Expected annual contract value exposed to predicted churn"],
            ["Expected Save Value", "Revenue at Risk × 30%", "Scenario value if intervention succeeds"],
            ["Net Expected Value", "Expected Save Value − cost", "Expected economic value after intervention cost"],
            ["Priority", "Economic exposure first", "Revenue at Risk / expected value drives portfolio ranking"],
        ],
        columns=["Assumption", "Value", "Purpose"],
    )

    st.dataframe(
        assumptions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Assumption": st.column_config.TextColumn("Assumption", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
            "Purpose": st.column_config.TextColumn("Purpose", width="large"),
        },
    )

    st.html(
        """
        <div class="governance-disclaimer">
            <b>Important:</b> the 30% intervention success rate and ₹25,000 intervention cost are
            explicit scenario assumptions. They are not learned causal treatment effects and should
            be replaced with measured intervention outcomes when production retention data becomes available.
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # AI GOVERNANCE
    # --------------------------------------------------------

    st.subheader("AI Governance & Guardrails")

    governance_df = pd.DataFrame(
        [
            ["Churn probability", "Calibrated XGBoost", "LLM cannot change the prediction"],
            ["Risk explanation", "SHAP", "Drivers are grounded in model output"],
            ["Business priority", "Decision Engine", "LLM cannot override economic ranking"],
            ["Recommended action", "Decision Engine", "LLM communicates the authoritative action"],
            ["Executive/customer narrative", "Qwen3:4B", "Communication only; no prediction authority"],
            ["Financial assumptions", "Business configuration", "Explicit and reviewable; not learned by the LLM"],
        ],
        columns=["Decision", "Authoritative Layer", "Control"],
    )

    st.dataframe(
        governance_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Decision": st.column_config.TextColumn("Decision", width="medium"),
            "Authoritative Layer": st.column_config.TextColumn("Authoritative Layer", width="medium"),
            "Control": st.column_config.TextColumn("Control", width="large"),
        },
    )

    st.write("")

    # --------------------------------------------------------
    # DEPLOYMENT READINESS
    # --------------------------------------------------------

    st.subheader("Deployment Readiness")

    readiness = [
        ("Production model artifacts", "Ready", "Calibrated XGBoost, preprocessor and business configuration are versioned in the models directory."),
        ("Deterministic inference", "Ready", "Production inference loads the frozen artifacts and reproduces model probabilities."),
        ("Decision engine", "Ready", "Risk, exposure, economics, recommendations and capacity selection are deterministic."),
        ("Explainability", "Ready", "SHAP explanations are generated from the production XGBoost model."),
        ("Local LLM layer", "Ready", "Qwen3:4B is used as a grounded communication layer through Ollama."),
        ("Outcome monitoring", "Monitor", "Requires real intervention outcomes to measure realised save rate, lift and ROI."),
        ("Model drift monitoring", "Monitor", "Production deployment should track feature drift, calibration and ranking performance."),
        ("Retraining / challenger process", "Monitor", "Establish retraining cadence and champion-challenger evaluation once production data accumulates."),
    ]

    readiness_rows = "".join(
        f"""
        <tr>
            <td>{html.escape(name)}</td>
            <td>
                <span class="governance-status {
                    'governance-status-ready' if status == 'Ready' else 'governance-status-monitor'
                }">
                    {html.escape(status)}
                </span>
            </td>
            <td>{html.escape(note)}</td>
        </tr>
        """
        for name, status, note in readiness
    )

    st.html(
        f"""
        <table class="governance-table">
            <thead>
                <tr>
                    <th>Control</th>
                    <th>Status</th>
                    <th>Readiness Note</th>
                </tr>
            </thead>
            <tbody>
                {readiness_rows}
            </tbody>
        </table>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # MONITORING ROADMAP
    # --------------------------------------------------------

    st.subheader("Production Monitoring Roadmap")

    st.markdown(
        """
        **1. Data quality** — monitor missingness, feature ranges, lifecycle coverage and schema changes.

        **2. Prediction quality** — monitor calibration, ROC-AUC / PR-AUC when labels mature, and probability distribution drift.

        **3. Decision quality** — track intervention acceptance, realised save rate, realised revenue saved and realised cost.

        **4. Business impact** — compare targeted customers with appropriate control groups to estimate incremental retention impact.

        **5. Model lifecycle** — retrain and challenge the champion when data drift or business performance warrants it.
        """
    )

    st.caption(
        "Realised retention lift and causal ROI require intervention and control-group outcome data; "
        "they cannot be established from the current synthetic development dataset alone."
    )


# ============================================================

# FOOTER
# ============================================================

st.html(
    """
    <div class="app-footer">

        <b>Retain-AI</b>
        &nbsp;•&nbsp;
        Customer Retention Intelligence

        <br>

        Calibrated XGBoost
        &nbsp;•&nbsp;
        Explainable AI
        &nbsp;•&nbsp;
        Business Decision Engine

    </div>
    """
)