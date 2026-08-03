import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="CDSS – Sepsis Early Warning",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; color:#1E3A8A; margin-bottom:0.1rem; }
    .badge-low  { background:#DEF7EC; color:#03543F; padding:0.6rem 1.2rem;
                  border-radius:25px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .badge-mod  { background:#FEF3C7; color:#92400E; padding:0.6rem 1.2rem;
                  border-radius:25px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .badge-high { background:#FDE8E8; color:#9B1C1C; padding:0.6rem 1.2rem;
                  border-radius:25px; font-weight:700; font-size:1.1rem; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# --- Load Pre-Trained Artifacts (Cached) ---
@st.cache_resource(show_spinner="Loading Clinical AI Model...")
def load_pipeline():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    with open(os.path.join(models_dir, 'imputer.pkl'),        'rb') as f: imp   = pickle.load(f)
    with open(os.path.join(models_dir, 'xgb_model.pkl'),     'rb') as f: xgb   = pickle.load(f)
    with open(os.path.join(models_dir, 'model_metadata.json'),'r', encoding='utf-8') as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(xgb)
    return imp, xgb, explainer, meta

imp, xgb_model, explainer, meta = load_pipeline()
engineered_cols = meta["engineered_cols"]
opt_threshold   = meta["opt_threshold"]
baseline_prev   = meta["baseline_prevalence"]
vitals          = meta["vitals"]
labs            = meta["labs"]

# ─── PRESET CONFIGURATION ──────────────────────────────────────────────────
# Preset values are grounded in actual training-set distributions so the model
# scores them at meaningfully different risk levels:
#
#   Case 1 (Normal)  → SIRS=0, HR≈70, SBP≈125  → model ~3-8%  (LOW)
#   Case 2 (Warning) → SIRS=1, HR≈95, SBP≈108  → model ~20-30% (MODERATE / near threshold)
#   Case 3 (Shock)   → SIRS=3, HR≈120,SBP≈82   → model ~60-90% (HIGH)
#
# Key: lab_orders_sum is kept at realistic multi-hour aggregate values (0.2–0.6)
#      not single-hour values which are out-of-distribution.
PRESETS = {
    "Custom Telemetry Input": dict(
        hr=80, sbp=118, map_val=82, temp=37.0, resp=16, o2=98,
        wbc=9.0, lac=1.4, glu=108, cre=1.0, plt=220, bun=15,
        age=58, gender=1,
        hr_span=5.0, map_span=5.0, temp_span=0.2,
        lab_orders_sum=0.2, lab_orders_mean=0.2
    ),
    # Calibrated: model scores ~22%  → LOW (below 23.4% threshold)
    # Physiology: SIRS=0, qSOFA=0, normal vitals
    "🟢 Case 1 – Normal Baseline": dict(
        hr=70, sbp=124, map_val=82, temp=36.6, resp=14, o2=98,
        wbc=5.5, lac=0.9, glu=92, cre=0.8, plt=270, bun=11,
        age=50, gender=1,
        hr_span=1.5, map_span=1.5, temp_span=0.08,
        lab_orders_sum=0.12, lab_orders_mean=0.12
    ),
    
    "🟠 Case 2 – Early Warning": dict(
        hr=80, sbp=104, map_val=75, temp=38.10, resp=19, o2=95,
        wbc=13.5, lac=1.9, glu=138, cre=1.3, plt=185, bun=24,
        age=58, gender=1,
        hr_span=10.0, map_span=8.0, temp_span=0.4,
        lab_orders_sum=0.35, lab_orders_mean=0.35
    ),
    # Calibrated: model scores ~74%  → HIGH (SIRS=4, qSOFA=2, septic shock)
    "🔴 Case 3 – Septic Shock": dict(
        hr=120, sbp=84, map_val=56, temp=38.9, resp=26, o2=91,
        wbc=17.5, lac=3.5, glu=195, cre=2.2, plt=110, bun=44,
        age=72, gender=0,
        hr_span=18.0, map_span=14.0, temp_span=0.7,
        lab_orders_sum=0.55, lab_orders_mean=0.55
    ),
}

# --- Header ---
st.markdown('<div class="main-title">🩺 Clinical Decision Support System — Early Sepsis Warning</div>',
            unsafe_allow_html=True)
st.caption("Educational CDSS | PhysioNet Sepsis Challenge | XGBoost + SHAP Pipeline")

# --- Sidebar Presets ---
st.sidebar.header("📋 Patient Case Presets")
preset_key = st.sidebar.radio("Select preset:", list(PRESETS.keys()))
p = PRESETS[preset_key]

# --- Input Panel ---
st.markdown("### 🎛️ Patient Telemetry Entry")
c1, c2 = st.columns(2)

with c1:
    st.subheader("🫀 Vital Signs")
    hr      = st.number_input("Heart Rate (bpm)",              20,  250, p["hr"])
    sbp     = st.number_input("Systolic BP (mmHg)",            40,  300, p["sbp"])
    map_val = st.number_input("Mean Arterial Pressure (mmHg)", 20,  200, p["map_val"])
    temp    = st.number_input("Body Temperature (°C)",         25.0,45.0,float(p["temp"]), step=0.1)
    resp    = st.number_input("Respiratory Rate (bpm)",         3,   80, p["resp"])
    o2sat   = st.number_input("O2 Saturation (%)",             50,  100, p["o2"])
    age     = st.number_input("Age (Years)",                   18,  100, p["age"])
    gender  = st.selectbox("Gender", [1, 0],
                           format_func=lambda x: "Male" if x == 1 else "Female",
                           index=0 if p["gender"] == 1 else 1)

with c2:
    st.subheader("🔬 Laboratory Panels")
    wbc        = st.number_input("WBC (×10⁹/L)",       0.5, 50.0, float(p["wbc"]),  step=0.5)
    lactate    = st.number_input("Lactate (mmol/L)",   0.1, 20.0, float(p["lac"]),  step=0.1)
    glucose    = st.number_input("Glucose (mg/dL)",     30,  600,  p["glu"])
    creatinine = st.number_input("Creatinine (mg/dL)", 0.1, 15.0, float(p["cre"]),  step=0.1)
    platelets  = st.number_input("Platelets (×10⁹/L)", 10,  800,  p["plt"])
    bun        = st.number_input("BUN (mg/dL)",          1,  200,  p["bun"])

st.markdown("---")

# --- Predict ---
if st.button("🚀 Estimate Sepsis Risk & Generate SHAP Explanation", type="primary", use_container_width=True):

    safe_sbp    = max(sbp, 1)
    shock_index = hr / safe_sbp

    sirs_temp  = 1 if (temp > 38.0 or temp < 36.0) else 0
    sirs_hr    = 1 if hr > 90 else 0
    sirs_resp  = 1 if resp > 20 else 0
    sirs_wbc   = 1 if (wbc > 12 or wbc < 4) else 0
    sirs_score = sirs_temp + sirs_hr + sirs_resp + sirs_wbc

    qsofa_score = (1 if resp >= 22 else 0) + (1 if sbp <= 100 else 0)

    # ── Build feature row matching training pipeline ──────────────────────
    # The model expects per-patient AGGREGATED (multi-hour) statistics.
    # We simulate them by:
    #   mean/min/max  → current single-hour reading
    #   std           → 0 (single snapshot)
    #   span features → preset span values (clinically plausible variability)
    #   lab_orders    → preset aggregate rate (not single-hour integer)
    row = {}
    for col, val in zip(vitals + labs,
                        [hr, o2sat, temp, sbp, map_val, resp,
                         wbc, lactate, glucose, creatinine, platelets, bun]):
        row[col + '_mean'] = val
        row[col + '_min']  = val
        row[col + '_max']  = val
        row[col + '_std']  = 0.0

    row['Age_first']    = age
    row['Gender_first'] = gender

    row['Shock_Index_mean'] = shock_index
    row['Shock_Index_max']  = shock_index
    row['Shock_Index_last'] = shock_index

    # Use realistic aggregate lab-order rate (not single-hour integer)
    row['Lab_Orders_This_Hour_sum']  = p["lab_orders_sum"]
    row['Lab_Orders_This_Hour_mean'] = p["lab_orders_mean"]
    row['Lab_Orders_This_Hour_max']  = p["lab_orders_mean"]

    row['SIRS_Score_mean']  = float(sirs_score)
    row['SIRS_Score_max']   = float(sirs_score)
    row['SIRS_Score_last']  = float(sirs_score)

    row['qSOFA_Score_mean']  = float(qsofa_score)
    row['qSOFA_Score_max']   = float(qsofa_score)
    row['qSOFA_Score_last']  = float(qsofa_score)

    row_df = pd.DataFrame([row])

    # Span features: simulate multi-hour physiologic variability from preset
    row_df['HR_span']   = float(p["hr_span"])
    row_df['MAP_span']  = float(p["map_span"])
    row_df['Temp_span'] = float(p["temp_span"])
    row_df['BUN_Creatinine_Ratio'] = bun / (creatinine + 1e-5)

    for col in engineered_cols:
        if col not in row_df.columns:
            row_df[col] = np.nan
    row_df = row_df[engineered_cols]

    input_imp = imp.transform(row_df)
    prob      = xgb_model.predict_proba(input_imp)[0, 1]
    prev_ratio = prob / baseline_prev

    # --- Result Cards ---
    st.markdown("### 📊 Risk Assessment Output")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sepsis Risk Probability", f"{prob*100:.1f}%",   f"{prev_ratio:.1f}× baseline")
    m2.metric("Shock Index (HR/SBP)",    f"{shock_index:.2f}", "Normal < 0.70")
    m3.metric("SIRS Score",              f"{sirs_score} / 4",
              "≥2 = Systemic Inflammation" if sirs_score >= 2 else "Low SIRS")
    m4.metric("qSOFA Score",             f"{qsofa_score} / 2",
              "≥2 = High Organ Risk" if qsofa_score >= 2 else "Low qSOFA")

    st.markdown("<br>", unsafe_allow_html=True)

    # Three-tier badge
    if prob >= opt_threshold:
        st.markdown(
            f'<div class="badge-high">⚠️ HIGH RISK ESTIMATE — '
            f'Probability {prob*100:.1f}% exceeds threshold {opt_threshold*100:.1f}%</div>',
            unsafe_allow_html=True)
    elif prob >= opt_threshold * 0.55:
        st.markdown(
            f'<div class="badge-mod">🟠 MODERATE RISK — '
            f'Probability {prob*100:.1f}% (threshold {opt_threshold*100:.1f}%)</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="badge-low">✅ LOW RISK ESTIMATE — '
            f'Probability {prob*100:.1f}% is below threshold {opt_threshold*100:.1f}%</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SHAP Explanation ---
    st.markdown("### 🔍 SHAP Feature Explanation")
    st.caption("Red bars increase risk · Blue bars decrease risk")

    shap_vals   = explainer(input_imp)
    single_shap = shap_vals[0]
    single_shap.feature_names = imp.get_feature_names_out(engineered_cols)

    fig, _ = plt.subplots(figsize=(10, 5))
    shap.plots.bar(single_shap, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.warning(
    "⚠️ **EDUCATIONAL DISCLAIMER** — This tool is for educational and presentation purposes only. "
    "It is not clinically validated and must not be used to guide real patient care or medical decisions."
)
