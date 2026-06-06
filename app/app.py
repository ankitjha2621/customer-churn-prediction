# ============================================================
#  CHURN INTELLIGENCE PLATFORM  —  app.py
#  Production-grade SaaS Analytics Dashboard
#  Stack: Streamlit · Plotly · Scikit-Learn · Joblib
# ============================================================

# ── 1. CONFIGURATION ────────────────────────────────────────
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import os
from datetime import datetime

st.set_page_config(
    page_title="ChurnIQ · Customer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH           = "models/churn_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

# ── 2. CUSTOM STYLING ────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

    /* ── Root tokens ────────────────────────────────────── */
    :root {
        --bg-base:        #080b14;
        --bg-surface:     #0d1120;
        --bg-glass:       rgba(255,255,255,0.04);
        --bg-glass-hover: rgba(255,255,255,0.07);
        --border:         rgba(255,255,255,0.08);
        --border-bright:  rgba(255,255,255,0.16);
        --accent-cyan:    #00e5ff;
        --accent-violet:  #8b5cf6;
        --accent-rose:    #f43f5e;
        --accent-amber:   #f59e0b;
        --accent-emerald: #10b981;
        --text-primary:   #f0f4ff;
        --text-secondary: #8892b0;
        --text-dim:       #4a5568;
        --gradient-a: linear-gradient(135deg, #00e5ff22 0%, #8b5cf622 100%);
        --gradient-b: linear-gradient(135deg, #f43f5e22 0%, #f59e0b22 100%);
        --gradient-c: linear-gradient(135deg, #10b98122 0%, #00e5ff22 100%);
        --glow-cyan:  0 0 40px rgba(0,229,255,0.15);
        --glow-rose:  0 0 40px rgba(244,63,94,0.20);
        --radius-lg:  16px;
        --radius-xl:  24px;
    }

    /* ── Global reset ───────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit chrome ──────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 0 2rem 4rem !important;
        max-width: 1400px !important;
    }

    /* ── Scrollbar ──────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }

    /* ── Sidebar ────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Glass card ─────────────────────────────────────── */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 28px 32px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: border-color .25s, box-shadow .25s;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: var(--border-bright);
        box-shadow: var(--glow-cyan);
    }
    .glass-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-a);
        opacity: 0;
        transition: opacity .3s;
        pointer-events: none;
    }
    .glass-card:hover::before { opacity: 1; }

    /* ── KPI card ───────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 22px 24px;
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
        transition: transform .2s, box-shadow .2s;
    }
    .kpi-card:hover { transform: translateY(-3px); box-shadow: var(--glow-cyan); }
    .kpi-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 36px;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-sub {
        font-size: 12px;
        color: var(--text-dim);
        margin-top: 6px;
    }
    .kpi-accent {
        position: absolute;
        top: 0; right: 0;
        width: 80px; height: 80px;
        border-radius: 0 var(--radius-lg) 0 100%;
        opacity: .12;
    }

    /* ── Hero section ───────────────────────────────────── */
    .hero-wrap {
        background: linear-gradient(135deg, #080b14 0%, #0f1630 50%, #080b14 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 48px 56px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(0,229,255,.12), transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent-cyan);
        font-weight: 600;
        margin-bottom: 12px;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(32px, 4vw, 52px);
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #f0f4ff 0%, #8892b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
    }
    .hero-sub {
        font-size: 16px;
        color: var(--text-secondary);
        font-weight: 300;
        max-width: 520px;
        line-height: 1.65;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,229,255,.1);
        border: 1px solid rgba(0,229,255,.25);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        color: var(--accent-cyan);
        font-weight: 500;
        margin-top: 24px;
    }

    /* ── Section headings ───────────────────────────────── */
    .section-heading {
        font-family: 'Syne', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-heading span.dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-cyan);
        display: inline-block;
        box-shadow: 0 0 8px var(--accent-cyan);
    }

    /* ── Risk badge ─────────────────────────────────────── */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        border-radius: 999px;
        font-family: 'Syne', sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: .5px;
    }
    .risk-low    { background: rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.4); color:#10b981; }
    .risk-medium { background: rgba(245,158,11,.15); border:1px solid rgba(245,158,11,.4); color:#f59e0b; }
    .risk-high   { background: rgba(244,63,94,.15);  border:1px solid rgba(244,63,94,.4);  color:#f43f5e; }

    /* ── Result card ────────────────────────────────────── */
    .result-card {
        border-radius: var(--radius-xl);
        padding: 40px;
        text-align: center;
        border: 1px solid var(--border);
        backdrop-filter: blur(24px);
        position: relative;
        overflow: hidden;
        animation: fadeSlideUp .5s ease both;
    }
    .result-churn {
        background: linear-gradient(135deg, rgba(244,63,94,.12) 0%, rgba(139,92,246,.08) 100%);
        box-shadow: var(--glow-rose);
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(16,185,129,.12) 0%, rgba(0,229,255,.08) 100%);
        box-shadow: var(--glow-cyan);
    }
    .result-headline {
        font-family: 'Syne', sans-serif;
        font-size: 28px;
        font-weight: 800;
        margin: 16px 0 8px;
    }
    .result-emoji { font-size: 52px; display: block; }

    /* ── Insight chip ───────────────────────────────────── */
    .insight-chip {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14px;
        color: var(--text-secondary);
        display: flex;
        align-items: flex-start;
        gap: 10px;
        transition: border-color .2s;
    }
    .insight-chip:hover { border-color: var(--border-bright); }
    .insight-chip .ic-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }

    /* ── Streamlit widget overrides ─────────────────────── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    .stSelectbox select,
    div[data-baseweb="select"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-baseweb="select"] * { color: var(--text-primary) !important; }
    .stSlider [data-baseweb="slider"] { accent-color: var(--accent-cyan); }

    /* Streamlit button */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
        color: #080b14 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        letter-spacing: .5px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        width: 100% !important;
        transition: opacity .2s, transform .2s !important;
    }
    .stButton > button:hover {
        opacity: .9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 32px rgba(0,229,255,.25) !important;
    }

    /* Label colour */
    label { color: var(--text-secondary) !important; font-size: 13px !important; }

    /* Dividers */
    hr { border-color: var(--border) !important; margin: 28px 0 !important; }

    /* ── Animations ─────────────────────────────────────── */
    @keyframes fadeSlideUp {
        from { opacity:0; transform:translateY(20px); }
        to   { opacity:1; transform:translateY(0);    }
    }
    @keyframes pulse-dot {
        0%,100% { box-shadow: 0 0 0 0 rgba(0,229,255,.6); }
        50%      { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
    }
    .live-dot {
        display:inline-block;
        width:8px; height:8px;
        border-radius:50%;
        background: var(--accent-cyan);
        animation: pulse-dot 2s infinite;
        vertical-align: middle;
        margin-right: 6px;
    }

    /* ── Footer ─────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 32px 0 8px;
        color: var(--text-dim);
        font-size: 12px;
        border-top: 1px solid var(--border);
        margin-top: 48px;
    }
    .footer a { color: var(--accent-cyan); text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ── 3. MODEL LOADER ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    mdl  = joblib.load(MODEL_PATH)           if os.path.exists(MODEL_PATH)           else None
    cols = joblib.load(FEATURE_COLUMNS_PATH) if os.path.exists(FEATURE_COLUMNS_PATH) else None
    return mdl, cols

model, feature_columns = load_model_and_features()

# ── 4. SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(135deg,#00e5ff,#8b5cf6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            ⚡ ChurnIQ
        </div>
        <div style="font-size:11px;color:#4a5568;letter-spacing:1.5px;
                    text-transform:uppercase;margin-top:4px;">
            Customer Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔮  Predict Churn", "📊  Analytics", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#4a5568;padding:8px 0;">
        <div style="margin-bottom:10px;">
            <span class="live-dot"></span>
            <span style="color:#8892b0;">Model Status</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model:
        st.success("✓ Model loaded & ready")
    else:
        st.error("⚠ Model not found")
        st.caption(f"Expected: `{MODEL_PATH}`")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#4a5568;line-height:1.8;">
        <div>🔒 &nbsp;Enterprise-grade security</div>
        <div>⚡ &nbsp;Real-time inference</div>
        <div>📡 &nbsp;REST API ready</div>
        <div>🌐 &nbsp;Multi-tenant support</div>
    </div>
    """, unsafe_allow_html=True)

# ── 5. HERO SECTION ──────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">⚡ AI-Powered Analytics Platform</div>
    <div class="hero-title">Customer Churn<br>Intelligence Hub</div>
    <div class="hero-sub">
        Predict, understand, and prevent customer churn with
        enterprise-grade machine learning. Turn data signals into
        retention strategies before it's too late.
    </div>
    <div class="hero-badge">
        <span class="live-dot"></span>
        Live Prediction Engine Active
    </div>
</div>
""", unsafe_allow_html=True)

# ── 6. KPI CARDS ─────────────────────────────────────────────
st.markdown('<div class="section-heading"><span class="dot"></span> Platform Overview</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

kpis = [
    ("Monthly Customers", "24,819", "+3.2% vs last month", "#00e5ff"),
    ("Avg Churn Rate", "6.4%", "↓ 1.1pp improvement",   "#8b5cf6"),
    ("Revenue at Risk", "$1.2M", "High-risk segment",    "#f43f5e"),
    ("Retention Score", "93.6", "Out of 100",            "#10b981"),
]

for col, (label, val, sub, color) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="background:linear-gradient(135deg,{color},{color}99);
                 -webkit-background-clip:text;background-clip:text;">{val}</div>
            <div class="kpi-sub">{sub}</div>
            <div class="kpi-accent" style="background:{color};"></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 7. ANALYTICS OVERVIEW CHARTS ─────────────────────────────
if nav in ["🏠  Dashboard", "📊  Analytics"]:
    st.markdown('<div class="section-heading"><span class="dot"></span> Analytics Overview</div>',
                unsafe_allow_html=True)

    ch_left, ch_right = st.columns([1.2, 1])

    with ch_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        churn_data   = [8.2, 7.8, 7.1, 6.9, 7.4, 6.5, 6.8, 6.2, 5.9, 6.1, 6.4, 6.4]
        retained     = [91.8, 92.2, 92.9, 93.1, 92.6, 93.5, 93.2, 93.8, 94.1, 93.9, 93.6, 93.6]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months, y=retained,
            name="Retained %", mode="lines+markers",
            line=dict(color="#00e5ff", width=3),
            marker=dict(size=6, color="#00e5ff"),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.06)"
        ))
        fig_trend.add_trace(go.Scatter(
            x=months, y=churn_data,
            name="Churn %", mode="lines+markers",
            line=dict(color="#f43f5e", width=2, dash="dot"),
            marker=dict(size=5, color="#f43f5e"),
        ))
        fig_trend.update_layout(
            title=dict(text="Monthly Churn vs Retention", font=dict(family="Syne",size=15,color="#f0f4ff")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
            margin=dict(l=10, r=10, t=40, b=10), height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[58, 28, 14],
            hole=.65,
            marker=dict(colors=["#10b981", "#f59e0b", "#f43f5e"],
                        line=dict(color="#080b14", width=3)),
            textfont=dict(family="DM Sans", size=13),
        )])
        fig_donut.update_layout(
            title=dict(text="Risk Distribution", font=dict(family="Syne",size=15,color="#f0f4ff")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=40, b=10), height=300,
            annotations=[dict(text="<b>24.8K</b>", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=20, color="#f0f4ff", family="Syne"))]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 8. PREDICTION SECTION ────────────────────────────────────
if nav in ["🏠  Dashboard", "🔮  Predict Churn"]:

    st.markdown('<div class="section-heading"><span class="dot"></span> Customer Churn Predictor</div>',
                unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1], gap="large")

    # ── 8a. INPUT FORM ───────────────────────────────────────
    with form_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Customer Profile**")
        st.markdown('<hr style="margin:12px 0 20px;">', unsafe_allow_html=True)

        # ── Row 1: numeric + gender
        col_a, col_b = st.columns(2)
        with col_a:
            tenure          = st.number_input("Tenure (months)", 0, 120, 12,
                                              help="Months the customer has been with the company")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with col_b:
            total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                              float(12 * 65), step=10.0)
            gender          = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: service & contract selects
        col_c, col_d = st.columns(2)
        with col_c:
            contract        = st.selectbox("Contract Type",
                                           ["Month-to-month", "One year", "Two year"])
            internet        = st.selectbox("Internet Service",
                                           ["DSL", "Fiber optic", "No"])
            multiple_lines  = st.selectbox("Multiple Lines",
                                           ["No", "Yes", "No phone service"])
        with col_d:
            payment         = st.selectbox("Payment Method",
                                           ["Electronic check", "Mailed check",
                                            "Bank transfer (automatic)",
                                            "Credit card (automatic)"])
            online_security = st.selectbox("Online Security",
                                           ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup",
                                           ["No", "Yes", "No internet service"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 3: add-on services
        col_e, col_f = st.columns(2)
        with col_e:
            device_protection = st.selectbox("Device Protection",
                                             ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support",
                                             ["No", "Yes", "No internet service"])
        with col_f:
            streaming_tv      = st.selectbox("Streaming TV",
                                             ["No", "Yes", "No internet service"])
            streaming_movies  = st.selectbox("Streaming Movies",
                                             ["No", "Yes", "No internet service"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 4: boolean flags
        col_g, col_h, col_i = st.columns(3)
        with col_g:
            senior     = st.checkbox("Senior Citizen")
            partner    = st.checkbox("Has Partner")
        with col_h:
            dependents = st.checkbox("Dependents")
            phone_svc  = st.checkbox("Phone Service", value=True)
        with col_i:
            paperless  = st.checkbox("Paperless Billing", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run Churn Prediction")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 8b. PREDICTION LOGIC + RESULTS ───────────────────────
    with result_col:
        if predict_btn:
            if not model:
                st.error("⚠️ Model not loaded. Ensure `models/churn_model.pkl` exists.")
            else:
                # ── Build raw input dict — IBM Telco feature names exactly ──
                raw = {
                    "gender":           gender,
                    "SeniorCitizen":    int(senior),
                    "Partner":          "Yes" if partner    else "No",
                    "Dependents":       "Yes" if dependents else "No",
                    "tenure":           tenure,
                    "PhoneService":     "Yes" if phone_svc  else "No",
                    "MultipleLines":    multiple_lines,
                    "InternetService":  internet,
                    "OnlineSecurity":   online_security,
                    "OnlineBackup":     online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport":      tech_support,
                    "StreamingTV":      streaming_tv,
                    "StreamingMovies":  streaming_movies,
                    "Contract":         contract,
                    "PaperlessBilling": "Yes" if paperless  else "No",
                    "PaymentMethod":    payment,
                    "MonthlyCharges":   monthly_charges,
                    "TotalCharges":     total_charges,
                }

                # ── Encode categoricals to match training encoding ──
                raw["gender"]          = 1 if raw["gender"]          == "Male" else 0
                raw["Partner"]         = 1 if raw["Partner"]         == "Yes"  else 0
                raw["Dependents"]      = 1 if raw["Dependents"]      == "Yes"  else 0
                raw["PhoneService"]    = 1 if raw["PhoneService"]    == "Yes"  else 0
                raw["PaperlessBilling"]= 1 if raw["PaperlessBilling"]== "Yes"  else 0

                ml_map = {"No": 0, "Yes": 1, "No phone service": 2}
                raw["MultipleLines"] = ml_map[raw["MultipleLines"]]

                raw["InternetService"] = {"DSL": 0, "Fiber optic": 1, "No": 2}[raw["InternetService"]]

                three_val = {"No": 0, "Yes": 1, "No internet service": 2}
                for feat in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                             "TechSupport", "StreamingTV", "StreamingMovies"]:
                    raw[feat] = three_val[raw[feat]]

                raw["Contract"] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[raw["Contract"]]

                raw["PaymentMethod"] = {
                    "Electronic check":          0,
                    "Mailed check":              1,
                    "Bank transfer (automatic)": 2,
                    "Credit card (automatic)":   3,
                }[raw["PaymentMethod"]]

                # ── Assemble DataFrame; reindex to trained feature order ──
                input_df = pd.DataFrame([raw])
                if feature_columns is not None:
                    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

                # ── Inference ──
                with st.spinner("Running inference…"):
                    time.sleep(0.6)
                    prediction = model.predict(input_df)[0]
                    churn_prob = float(model.predict_proba(input_df)[0][1])

                confidence = max(churn_prob, 1 - churn_prob) * 100

                # Risk category
                if churn_prob < 0.35:
                    risk_label, risk_cls, risk_icon = "Low Risk",    "risk-low",    "🟢"
                elif churn_prob < 0.65:
                    risk_label, risk_cls, risk_icon = "Medium Risk", "risk-medium", "🟡"
                else:
                    risk_label, risk_cls, risk_icon = "High Risk",   "risk-high",   "🔴"

                card_cls = "result-churn" if prediction else "result-safe"
                headline = ("Customer Likely to Churn" if prediction
                            else "Customer Likely to Stay")
                emoji    = "⚠️" if prediction else "✅"
                clr      = "#f43f5e" if prediction else "#10b981"

                # ── Result card
                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <span class="result-emoji">{emoji}</span>
                    <div class="result-headline" style="color:{clr};">{headline}</div>
                    <span class="risk-badge {risk_cls}">{risk_icon} {risk_label}</span>
                    <div style="display:flex;justify-content:center;gap:40px;margin-top:24px;">
                        <div style="text-align:center;">
                            <div style="font-size:11px;letter-spacing:1.2px;color:#8892b0;text-transform:uppercase;">Churn Probability</div>
                            <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:{clr};">
                                {churn_prob*100:.1f}%
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:11px;letter-spacing:1.2px;color:#8892b0;text-transform:uppercase;">Confidence</div>
                            <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#00e5ff;">
                                {confidence:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Gauge chart
                st.markdown('<div class="glass-card" style="padding:20px;">', unsafe_allow_html=True)
                gauge_color = "#10b981" if churn_prob < .35 else ("#f59e0b" if churn_prob < .65 else "#f43f5e")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(churn_prob * 100, 1),
                    number=dict(suffix="%", font=dict(family="Syne", size=36, color="#f0f4ff")),
                    delta=dict(reference=50, valueformat=".1f",
                               increasing=dict(color="#f43f5e"),
                               decreasing=dict(color="#10b981")),
                    gauge=dict(
                        axis=dict(range=[0,100], tickcolor="#4a5568",
                                  tickfont=dict(color="#8892b0", size=11)),
                        bar=dict(color=gauge_color, thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.1)",
                        steps=[
                            dict(range=[0,35],  color="rgba(16,185,129,.12)"),
                            dict(range=[35,65], color="rgba(245,158,11,.12)"),
                            dict(range=[65,100],color="rgba(244,63,94,.12)"),
                        ],
                        threshold=dict(line=dict(color="#f0f4ff",width=2),
                                       thickness=0.75, value=churn_prob*100)
                    ),
                    title=dict(text="Churn Risk Meter", font=dict(family="Syne",size=14,color="#8892b0"))
                ))
                fig_gauge.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20,r=20,t=30,b=10), height=240,
                    font=dict(color="#8892b0")
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar":False})
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Persist result for panels below ──
                st.session_state["last_result"] = {
                    "churn_prob":       churn_prob,
                    "prediction":       prediction,
                    "confidence":       confidence,
                    "risk_label":       risk_label,
                    "tenure":           tenure,
                    "monthly_charges":  monthly_charges,
                    "contract":         contract,
                    "internet":         internet,
                    "tech_support":     tech_support,
                    "online_security":  online_security,
                    "streaming_tv":     streaming_tv,
                    "streaming_movies": streaming_movies,
                }

        else:
            # Placeholder state
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:60px 32px;">
                <div style="font-size:56px;">🎯</div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                            margin-top:16px;color:#f0f4ff;">
                    Ready to Predict
                </div>
                <div style="color:#8892b0;font-size:14px;margin-top:10px;line-height:1.6;">
                    Fill in the customer profile on the left<br>
                    and click <strong style="color:#00e5ff;">Run Churn Prediction</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 9. BUSINESS INSIGHTS + RECOMMENDATIONS ───────────────
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        cp  = res["churn_prob"]

        st.markdown("<br>", unsafe_allow_html=True)
        ins_col, rec_col = st.columns(2, gap="large")

        with ins_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading"><span class="dot"></span> Business Insights</div>',
                        unsafe_allow_html=True)

            # ── Dynamic insights driven by IBM Telco signals ──
            insights = []
            if res["tenure"] < 6:
                insights.append(("📅", "Early-stage customer (< 6 months) — historically higher churn window."))
            if res["monthly_charges"] > 80:
                insights.append(("💸", "Above-average billing ($80+) correlates with elevated churn sensitivity."))
            if res["contract"] == "Month-to-month":
                insights.append(("📋", "Month-to-month contracts churn 3× more than annual plans."))
            if res.get("internet") == "Fiber optic":
                insights.append(("🌐", "Fiber optic customers show higher churn — often price-sensitive."))
            if res.get("tech_support") == "No" and res.get("internet") != "No":
                insights.append(("🛠", "No Tech Support subscription increases churn likelihood significantly."))
            if res.get("online_security") == "No" and res.get("internet") != "No":
                insights.append(("🔒", "No Online Security service correlates with elevated churn risk."))
            if cp > 0.6:
                insights.append(("🚨", "High-probability churn — intervention within 7 days is critical."))
            if not insights:
                insights = [
                    ("🟢", "Customer profile looks healthy — low churn indicators."),
                    ("📈", "Continue engagement strategy to maintain retention."),
                ]

            for icon, text in insights[:5]:
                st.markdown(f"""
                <div class="insight-chip">
                    <span class="ic-icon">{icon}</span>
                    <span>{text}</span>
                </div>
                """, unsafe_allow_html=True)

            # Probability bar
            bar_pct = int(cp * 100)
            bar_clr = "#10b981" if cp < .35 else ("#f59e0b" if cp < .65 else "#f43f5e")
            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;
                            color:#8892b0;margin-bottom:6px;">
                    <span>Churn Probability</span><span>{bar_pct}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">
                    <div style="width:{bar_pct}%;height:100%;border-radius:999px;
                                background:linear-gradient(90deg,{bar_clr}88,{bar_clr});
                                transition:width .8s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rec_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading"><span class="dot"></span> Recommendations</div>',
                        unsafe_allow_html=True)

            if cp >= 0.65:
                recs = [
                    ("🎁", "Urgent Retention Offer", "Deploy a personalised discount (15–20%) within 48 hours."),
                    ("📞", "Direct Outreach", "Assign a dedicated CSM for immediate relationship repair."),
                    ("🔄", "Contract Upgrade", "Offer a 3-month free trial of an annual plan."),
                    ("🛠", "Support Audit", "Escalate open tickets — unresolved issues drive exits."),
                ]
            elif cp >= 0.35:
                recs = [
                    ("💌", "Engagement Campaign", "Trigger an email nurture sequence with product tips."),
                    ("⭐", "Loyalty Programme", "Invite to loyalty tier for early feature access."),
                    ("📊", "Health Check", "Schedule a quarterly business review call."),
                ]
            else:
                recs = [
                    ("🚀", "Upsell Opportunity", "Customer is healthy — ideal candidate for plan upgrade."),
                    ("🤝", "Referral Programme", "High-satisfaction customers convert referrals 4× better."),
                    ("📣", "Case Study", "Engage as a potential brand advocate or testimonial."),
                ]

            for icon, title, desc in recs:
                st.markdown(f"""
                <div class="insight-chip" style="flex-direction:column;gap:4px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:16px;">{icon}</span>
                        <strong style="color:#f0f4ff;font-size:13px;">{title}</strong>
                    </div>
                    <div style="padding-left:24px;font-size:13px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ── 10. RISK DISTRIBUTION SCATTER ────────────────────────────
if nav in ["🏠  Dashboard", "📊  Analytics"]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span class="dot"></span> Cohort Risk Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    np.random.seed(42)
    n = 300
    ten  = np.random.randint(0, 72, n)
    mchg = np.random.uniform(20, 120, n)
    prob = np.clip(0.9 - 0.012*ten + 0.004*mchg + np.random.normal(0,.15,n), 0, 1)
    risk_cat = np.where(prob < .35, "Low Risk", np.where(prob < .65, "Medium Risk", "High Risk"))
    color_map = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#f43f5e"}

    df_scatter = pd.DataFrame({"Tenure (months)": ten,
                                "Monthly Charges ($)": mchg,
                                "Churn Probability": prob,
                                "Risk": risk_cat})

    fig_scatter = px.scatter(
        df_scatter, x="Tenure (months)", y="Monthly Charges ($)",
        color="Risk", size="Churn Probability",
        color_discrete_map=color_map,
        size_max=18, opacity=0.8,
        hover_data={"Churn Probability": ":.1%"},
    )
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892b0", family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=""),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10,r=10,t=10,b=10), height=340,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── 11. CUSTOMER SUMMARY STRIP ────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-heading"><span class="dot"></span> Recent High-Risk Alerts</div>',
            unsafe_allow_html=True)

alert_customers = [
    ("C-00291", "Sarah Mitchell",   "87%", "#f43f5e", "Month-to-month", "$98.40"),
    ("C-00847", "James Okonkwo",    "74%", "#f43f5e", "Month-to-month", "$112.75"),
    ("C-01204", "Liu Wei",          "61%", "#f59e0b", "One year",       "$76.20"),
    ("C-02019", "Priya Krishnaswamy","58%","#f59e0b", "Month-to-month", "$88.50"),
]

ac1, ac2, ac3, ac4 = st.columns(4)
for col, (cid, name, prob, clr, contract, charge) in zip([ac1, ac2, ac3, ac4], alert_customers):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="padding:18px 20px;">
            <div style="font-size:10px;color:#4a5568;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">{cid}</div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:#f0f4ff;margin-bottom:10px;">{name}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:11px;color:#8892b0;">Churn Risk</span>
                <span style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{clr};">{prob}</span>
            </div>
            <div style="font-size:11px;color:#4a5568;">{contract} · {charge}/mo</div>
        </div>
        """, unsafe_allow_html=True)

# ── 12. ABOUT PAGE ────────────────────────────────────────────
if nav == "ℹ️  About":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width:680px;">
        <div class="section-heading"><span class="dot"></span> About ChurnIQ</div>
        <p style="color:#8892b0;line-height:1.8;font-size:15px;">
            ChurnIQ is an enterprise-grade customer churn prediction platform powered by
            machine learning. It enables customer success teams to identify at-risk customers
            before they leave, prioritise intervention, and automate personalised retention workflows.
        </p>
        <hr>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;">
            <div>
                <div style="font-size:11px;letter-spacing:1.2px;color:#4a5568;text-transform:uppercase;margin-bottom:8px;">Stack</div>
                <div style="color:#8892b0;font-size:14px;line-height:2;">
                    Python 3.10+<br>Scikit-Learn<br>Streamlit 1.x<br>Plotly 5.x<br>Joblib
                </div>
            </div>
            <div>
                <div style="font-size:11px;letter-spacing:1.2px;color:#4a5568;text-transform:uppercase;margin-bottom:8px;">Model Info</div>
                <div style="color:#8892b0;font-size:14px;line-height:2;">
                    Dataset: IBM Telco Customer Churn<br>
                    Features: 19 telecom attributes<br>
                    Threshold: 0.50 probability<br>
                    Output: Binary + probability
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 13. FOOTER ────────────────────────────────────────────────
year = datetime.now().year
st.markdown(f"""
<div class="footer">
    <div style="margin-bottom:8px;">
        <span style="font-family:'Syne',sans-serif;font-weight:700;
                     background:linear-gradient(135deg,#00e5ff,#8b5cf6);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text;">⚡ ChurnIQ</span>
        &nbsp;·&nbsp; Customer Intelligence Platform
    </div>
    <div>© {year} ChurnIQ · Built with Streamlit &amp; Plotly · AI-powered retention analytics</div>
</div>
""", unsafe_allow_html=True)