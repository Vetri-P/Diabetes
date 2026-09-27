"""
========================================================================================
Diabetes EDL Clinical Decision Support System — Interactive Streamlit App
Replicating the Ensemble Deep Learning (EDL) Framework:
Al Reshan et al., "An Innovative Ensemble Deep Learning Clinical Decision Support System
for Diabetes Prediction," IEEE Access, vol. 12, 2024.
========================================================================================
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes EDL Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------------
# Custom CSS Styling (Rich Dark-Mode Medical/AI Aesthetics)
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header hero styling */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(26, 38, 64, 0.9) 50%, rgba(13, 27, 42, 0.95) 100%);
        border: 1px solid rgba(0, 210, 190, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #00d2be;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }

    /* Pipeline Step Cards */
    .pipe-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 18px 0;
        justify-content: space-between;
    }
    .pipe-step {
        flex: 1 1 130px;
        background: rgba(19, 30, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .pipe-step:hover {
        border-color: #00d2be;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 210, 190, 0.15);
    }
    .pipe-num {
        background: #00d2be;
        color: #06111f;
        font-size: 11px;
        font-weight: 800;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 6px;
    }
    .pipe-title {
        color: #f1f5f9;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .pipe-sub {
        color: #94a3b8;
        font-size: 10.5px;
    }

    /* Model Group Badges */
    .badge-dl-base {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-dl-stack {
        background: rgba(249, 115, 22, 0.15);
        border: 1px solid rgba(249, 115, 22, 0.4);
        color: #fb923c;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Result Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        color: #34d399;
    }
    .risk-banner-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 16px;
        color: #fbbf24;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Paths and Cache Loaders
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
if not os.path.exists(ARTIFACT_DIR):
    ARTIFACT_DIR = BASE_DIR

@st.cache_resource
def load_edl_system(dataset_key):
    """Loads all Base DL models (ANN, LSTM, CNN) + Meta-Level Stacking models (Stack-ANN, Stack-LSTM, Stack-CNN)"""
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, f"{dataset_key}_scaler.joblib"))
    with open(os.path.join(ARTIFACT_DIR, f"{dataset_key}_meta.json"), "r") as f:
        meta = json.load(f)

    # Base DL Models
    base_models = {
        "ANN": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_ann.keras")),
        "LSTM": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_lstm.keras")),
        "CNN": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_cnn.keras")),
    }

    # Meta-Level Stacking Models
    meta_models = {
        "Stack-ANN": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_stack_ann.keras")),
        "Stack-LSTM": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_stack_lstm.keras")),
        "Stack-CNN": tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, f"{dataset_key}_stack_cnn.keras")),
    }

    encoders = None
    enc_path = os.path.join(ARTIFACT_DIR, f"{dataset_key}_label_encoders.joblib")
    if os.path.exists(enc_path):
        encoders = joblib.load(enc_path)

    return scaler, meta, base_models, meta_models, encoders

def predict_edl(scaler, meta, base_models, meta_models, input_dict: dict, selected_meta="Consensus Ensemble"):
    """Runs complete 2-Tier EDL inference: Base Learners -> Out-of-fold meta representation -> Meta Stacking Learners"""
    all_cols = meta["all_feature_columns"]
    medians = meta["all_feature_medians"]
    
    # Construct complete feature row and scale with fitted MinMax scaler
    full_row = {c: input_dict.get(c, medians.get(c, 0.0)) for c in all_cols}
    df_full = pd.DataFrame([full_row], columns=all_cols)
    scaled_full = pd.DataFrame(scaler.transform(df_full), columns=all_cols)
    
    X_selected = scaled_full[meta["selected_features"]].values.astype("float32")
    n_features = len(meta["selected_features"])
    X_seq = X_selected.reshape(-1, n_features, 1)

    binary = meta["binary"]

    # 1. Tier 1: Base-Level DL Predictions
    raw_p_ann = base_models["ANN"].predict(X_selected, verbose=0)
    raw_p_lstm = base_models["LSTM"].predict(X_seq, verbose=0)
    raw_p_cnn = base_models["CNN"].predict(X_seq, verbose=0)

    def format_p(raw):
        if binary:
            p = raw.reshape(-1)
            return np.column_stack([1.0 - p, p])
        return raw

    p_ann = format_p(raw_p_ann)[0]
    p_lstm = format_p(raw_p_lstm)[0]
    p_cnn = format_p(raw_p_cnn)[0]

    base_probas = {
        "ANN (Tabular)": p_ann,
        "LSTM (Sequential)": p_lstm,
        "CNN (Structural)": p_cnn
    }

    # 2. Tier 2: Meta-Level Feature Vector
    meta_vec = np.hstack([p_ann, p_lstm, p_cnn]).reshape(1, -1).astype("float32")
    meta_dim = meta_vec.shape[1]
    meta_vec_seq = meta_vec.reshape(1, meta_dim, 1)

    # 3. Meta-Level Stacking Predictions
    raw_sa = meta_models["Stack-ANN"].predict(meta_vec, verbose=0)
    raw_sl = meta_models["Stack-LSTM"].predict(meta_vec_seq, verbose=0)
    raw_sc = meta_models["Stack-CNN"].predict(meta_vec_seq, verbose=0)

    p_stack_ann = format_p(raw_sa)[0]
    p_stack_lstm = format_p(raw_sl)[0]
    p_stack_cnn = format_p(raw_sc)[0]
    p_consensus = (p_stack_ann + p_stack_lstm + p_stack_cnn) / 3.0

    meta_probas = {
        "Stack-ANN": p_stack_ann,
        "Stack-LSTM": p_stack_lstm,
        "Stack-CNN": p_stack_cnn,
        "Consensus Ensemble": p_consensus
    }

    chosen_proba = meta_probas.get(selected_meta, p_consensus)
    chosen_pred = int(np.argmax(chosen_proba))

    return chosen_pred, chosen_proba, base_probas, meta_probas

# --------------------------------------------------------------------------------------
# Sidebar Navigation & Settings
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="font-size:28px;">🩺</div>
            <div>
                <h3 style="margin:0; font-size:18px; font-weight:800; color:#ffffff;">Diabetes EDL</h3>
                <span style="font-size:11px; color:#00d2be; font-weight:600;">CLINICAL DECISION SUPPORT</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    dataset_option = st.radio(
        "Select Dataset & Clinical Target",
        ["BRFSS 2015 — 3-Class Diabetes Risk", "Readmission — 30-Day Hospital Risk"],
        index=0
    )
    dataset_key = "brfss" if dataset_option.startswith("BRFSS") else "readmit"
    
    st.markdown("---")
    st.subheader("⚡ Meta Stacking Engine")
    meta_engine_choice = st.selectbox(
        "Active Stacking Meta-Model",
        ["Consensus Ensemble (All 3)", "Stack-ANN", "Stack-LSTM", "Stack-CNN"],
        index=0,
        help="Select which meta-learner in Tier 2 drives the final clinical prediction."
    )

    st.markdown("---")
    st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px;">
            <div style="font-size:11px; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:4px;">Reference Paper</div>
            <div style="font-size:12px; color:#cbd5e1; line-height:1.4;">
                <b>Al Reshan et al.</b>, <i>IEEE Access</i>, vol. 12, 2024.<br>
                <span style="color:#94a3b8; font-size:11px;">Ensemble Deep Learning framework with base learners + meta stacking.</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption("⚠️ Research & Educational Decision Support Tool. Not for direct diagnosis.")

# Load models and metadata
scaler, meta, base_models, meta_models, encoders = load_edl_system(dataset_key)
class_names = meta["class_names"]

# --------------------------------------------------------------------------------------
# Hero Banner & Proposed System Visual Pipeline
# --------------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-subtitle">Ensemble Deep Learning (EDL) Clinical Decision Support System</div>
    <div class="hero-title">
        🩺 {dataset_option}
    </div>
    <p class="hero-desc">
        Faithful implementation of the multi-tier EDL framework from <b>Al Reshan et al. (IEEE Access, 2024)</b>:
        integrating tabular ANN, sequential LSTM, and structural CNN base learners into meta-level stacking models.
    </p>
</div>
""", unsafe_allow_html=True)

# Visual Pipeline Cards (Exact replica of user's diagram)
st.markdown("""
<div class="pipe-container">
    <div class="pipe-step">
        <div class="pipe-num">1</div>
        <div class="pipe-title">Diabetes Dataset</div>
        <div class="pipe-sub">BRFSS 2015 / Readmit</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">2</div>
        <div class="pipe-title">Preprocessing</div>
        <div class="pipe-sub">Clean, Impute, MinMax</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">3</div>
        <div class="pipe-title">Feature Selection</div>
        <div class="pipe-sub">Extra Trees Classifier</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">4</div>
        <div class="pipe-title">80/20 Split</div>
        <div class="pipe-sub">Stratified Train / Test</div>
    </div>
    <div class="pipe-step" style="border-color: rgba(59, 130, 246, 0.4); background: rgba(59, 130, 246, 0.08);">
        <div class="pipe-num" style="background:#3b82f6; color:#ffffff;">5</div>
        <div class="pipe-title" style="color:#60a5fa;">DL Base Models</div>
        <div class="pipe-sub">ANN • LSTM • CNN</div>
    </div>
    <div class="pipe-step" style="border-color: rgba(249, 115, 22, 0.4); background: rgba(249, 115, 22, 0.08);">
        <div class="pipe-num" style="background:#f97316; color:#ffffff;">6</div>
        <div class="pipe-title" style="color:#fb923c;">Meta Stacking</div>
        <div class="pipe-sub">Stack-ANN • LSTM • CNN</div>
    </div>
    <div class="pipe-step" style="border-color: #00d2be; background: rgba(0, 210, 190, 0.08);">
        <div class="pipe-num">7</div>
        <div class="pipe-title" style="color:#00d2be;">Final Prediction</div>
        <div class="pipe-sub">Clinical Risk Output</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Main Application Tabs
# --------------------------------------------------------------------------------------
tab_predict, tab_performance, tab_flowchart, tab_batch, tab_about = st.tabs([
    "🔮 Patient Risk Predictor",
    "📊 Model Performance & Metrics",
    "🗺️ Proposed System Architecture",
    "📁 Batch Patient Screening",
    "📖 Methodology & Equations"
])

# ======================================================================================
# TAB 1: Real-time Patient Risk Predictor
# ======================================================================================
with tab_predict:
    st.markdown("### 👤 Patient Clinical Profile Entry")
    
    # Preset quick-fills
    p_preset = st.selectbox(
        "⚡ Quick-Fill Patient Preset Profile:",
        ["Custom Patient", "Preset 1: Healthy Young Adult (Low Risk)", "Preset 2: Pre-Diabetic Indicators (Moderate Risk)", "Preset 3: High Risk Patient (Chronic/Comorbid)"]
    )
    
    input_dict = {}

    if dataset_key == "brfss":
        # BRFSS 2015 3-class Diabetes Risk
        c1, c2, c3 = st.columns(3)
        
        default_bmi = 26.5
        default_age = 7
        default_inc = 6
        default_gen = 2
        default_phys = 0
        default_ment = 0
        default_bp = 0
        default_chol = 0
        default_walk = 0
        default_smoke = 0

        if "Healthy" in p_preset:
            default_bmi = 22.0
            default_age = 3
            default_inc = 7
            default_gen = 1
            default_phys = 0
            default_ment = 0
            default_bp = 0
            default_chol = 0
            default_walk = 0
            default_smoke = 0
        elif "Pre-Diabetic" in p_preset:
            default_bmi = 29.5
            default_age = 8
            default_inc = 5
            default_gen = 3
            default_phys = 4
            default_ment = 2
            default_bp = 1
            default_chol = 1
            default_walk = 0
            default_smoke = 0
        elif "High Risk" in p_preset:
            default_bmi = 36.5
            default_age = 11
            default_inc = 3
            default_gen = 5
            default_phys = 15
            default_ment = 10
            default_bp = 1
            default_chol = 1
            default_walk = 1
            default_smoke = 1

        with c1:
            st.markdown("##### 🧬 Vitals & Demographics")
            bmi_val = st.number_input("BMI (Body Mass Index)", min_value=12.0, max_value=98.0, value=float(default_bmi), step=0.5)
            input_dict["BMI"] = bmi_val
            
            # Dynamic BMI classification indicator
            if bmi_val < 18.5:
                st.caption("BMI Status: 🔵 Underweight (< 18.5)")
            elif bmi_val < 25.0:
                st.caption("BMI Status: 🟢 Normal weight (18.5 - 24.9)")
            elif bmi_val < 30.0:
                st.caption("BMI Status: 🟡 Overweight (25.0 - 29.9)")
            else:
                st.caption("BMI Status: 🔴 Obese (≥ 30.0)")

            age_labels = {
                1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
                7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
            }
            age_sel = st.selectbox("Age Group", list(age_labels.keys()), index=default_age - 1, format_func=lambda k: age_labels[k])
            input_dict["Age"] = age_sel

            sex_sel = st.radio("Sex", ["Female", "Male"], index=1, horizontal=True)
            input_dict["Sex"] = 1 if sex_sel == "Male" else 0

            income_labels = {
                1: "< $10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
                5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: "$75k+"
            }
            income_sel = st.selectbox("Income Bracket", list(income_labels.keys()), index=default_inc - 1, format_func=lambda k: income_labels[k])
            input_dict["Income"] = income_sel

        with c2:
            st.markdown("##### 🩺 Health Perception & Vitals")
            gen_labels = {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
            gen_sel = st.selectbox("General Health Rating", list(gen_labels.keys()), index=default_gen - 1, format_func=lambda k: gen_labels[k])
            input_dict["GenHlth"] = gen_sel

            input_dict["PhysHlth"] = st.slider("Physically Unwell Days (past 30 days)", 0, 30, int(default_phys))
            input_dict["MentHlth"] = st.slider("Mentally Unwell Days (past 30 days)", 0, 30, int(default_ment))
            
            diff_walk = st.radio("Difficulty Walking or Climbing Stairs?", ["No", "Yes"], index=default_walk, horizontal=True)
            input_dict["DiffWalk"] = 1 if diff_walk == "Yes" else 0

            edu_labels = {1: "Never attended", 2: "Elementary", 3: "Some high school", 4: "High school graduate", 5: "Some college", 6: "College graduate"}
            edu_sel = st.selectbox("Education Level", list(edu_labels.keys()), index=4, format_func=lambda k: edu_labels[k])
            input_dict["Education"] = edu_sel

        with c3:
            st.markdown("##### 🥗 Lifestyle & Risk Factors")
            bp = st.radio("Diagnosed High Blood Pressure?", ["No", "Yes"], index=default_bp, horizontal=True)
            input_dict["HighBP"] = 1 if bp == "Yes" else 0

            chol = st.radio("Diagnosed High Cholesterol?", ["No", "Yes"], index=default_chol, horizontal=True)
            input_dict["HighChol"] = 1 if chol == "Yes" else 0

            smoker = st.radio("Smoked 100+ Cigarettes in Lifetime?", ["No", "Yes"], index=default_smoke, horizontal=True)
            input_dict["Smoker"] = 1 if smoker == "Yes" else 0

            act = st.radio("Physical Activity (Past 30 Days)?", ["No", "Yes"], index=1, horizontal=True)
            input_dict["PhysActivity"] = 1 if act == "Yes" else 0

            fruit = st.radio("Consumes Fruit ≥1 Time/Day?", ["No", "Yes"], index=1, horizontal=True)
            input_dict["Fruits"] = 1 if fruit == "Yes" else 0

            veg = st.radio("Consumes Vegetables ≥1 Time/Day?", ["No", "Yes"], index=1, horizontal=True)
            input_dict["Veggies"] = 1 if veg == "Yes" else 0

    else:
        # Diabetic Patient Hospital Readmission (<30 days)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🏥 Inpatient Clinical Vitals")
            input_dict["time_in_hospital"] = st.number_input("Time in Hospital (Days)", 1, 14, 4)
            input_dict["num_lab_procedures"] = st.number_input("# Lab Procedures Performed", 0, 150, 43)
            input_dict["num_medications"] = st.number_input("# Medications Administered", 0, 90, 16)
            input_dict["num_procedures"] = st.number_input("# Non-Lab Procedures", 0, 10, 1)
            input_dict["number_diagnoses"] = st.number_input("# Total Diagnoses Recorded", 1, 20, 8)

        with c2:
            st.markdown("##### 📊 Historical Encounters")
            input_dict["number_inpatient"] = st.number_input("# Prior Inpatient Admissions (Past Year)", 0, 25, 0)
            input_dict["number_emergency"] = st.number_input("# Prior Emergency Room Visits", 0, 80, 0)
            input_dict["number_outpatient"] = st.number_input("# Prior Outpatient Visits", 0, 45, 0)

            if encoders and "age" in encoders:
                age_keys = list(encoders["age"].keys())
                age_sel = st.selectbox("Patient Age Bracket", age_keys, index=min(6, len(age_keys)-1))
                input_dict["age"] = encoders["age"][age_sel]
            else:
                input_dict["age"] = 6

            if encoders and "race" in encoders:
                race_keys = [k for k in encoders["race"].keys() if isinstance(k, str) and k != "nan"]
                race_sel = st.selectbox("Race / Ethnicity", race_keys, index=min(2, len(race_keys)-1))
                input_dict["race"] = encoders["race"][race_sel]

        with c3:
            st.markdown("##### 💊 Medications & Admission Source")
            if encoders and "insulin" in encoders:
                ins_keys = list(encoders["insulin"].keys())
                ins_sel = st.selectbox("Insulin Adjustment", ins_keys, index=min(1, len(ins_keys)-1))
                input_dict["insulin"] = encoders["insulin"][ins_sel]

            if encoders and "A1Cresult" in encoders:
                a1c_keys = [k for k in encoders["A1Cresult"].keys() if isinstance(k, str) and k != "nan"]
                a1c_sel = st.selectbox("HbA1c Lab Test Result", a1c_keys + ["None / Not Tested"], index=0)
                input_dict["A1Cresult"] = encoders["A1Cresult"].get(a1c_sel, 3)

            input_dict["discharge_disposition_id"] = st.number_input("Discharge Code (1 = Routine Home)", 1, 30, 1)
            input_dict["admission_type_id"] = st.number_input("Admission Type Code", 1, 8, 1)
            input_dict["admission_source_id"] = st.number_input("Admission Source Code", 1, 25, 7)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Run EDL Ensemble Stacking Prediction", type="primary", use_container_width=True)

    if predict_btn:
        active_engine = "Consensus Ensemble" if meta_engine_choice.startswith("Consensus") else meta_engine_choice
        pred_idx, chosen_proba, base_probas, meta_probas = predict_edl(
            scaler, meta, base_models, meta_models, input_dict, selected_meta=active_engine
        )
        
        predicted_class_name = class_names[pred_idx]
        confidence_pct = chosen_proba[pred_idx] * 100.0

        st.markdown("---")
        st.subheader("🎯 Diagnostic Evaluation & Stacking Inference Results")

        res_col1, res_col2 = st.columns([1.1, 1.9])

        with res_col1:
            st.markdown(f"#### 🏆 {active_engine} Decision")
            
            # Risk coloring
            if pred_idx == 0:
                banner_cls = "risk-banner-low"
                status_icon = "🟢"
                risk_tag = "LOW RISK"
            elif pred_idx == 1 and not meta["binary"]:
                banner_cls = "risk-banner-med"
                status_icon = "🟡"
                risk_tag = "ELEVATED RISK"
            else:
                banner_cls = "risk-banner-high"
                status_icon = "🔴"
                risk_tag = "HIGH RISK / POSITIVE"

            st.markdown(f"""
            <div class="{banner_cls}">
                <div style="font-size:12px; font-weight:800; letter-spacing:1px; text-transform:uppercase;">Predicted Diagnosis</div>
                <div style="font-size:24px; font-weight:800; margin:6px 0;">{status_icon} {predicted_class_name}</div>
                <div style="font-size:14px; font-weight:600;">Confidence: <b>{confidence_pct:.1f}%</b> ({risk_tag})</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Tier-2 Probability Distribution:")
            for i, c_name in enumerate(class_names):
                pct = float(chosen_proba[i])
                st.write(f"**{c_name}**: `{pct*100:.1f}%`")
                st.progress(pct)

        with res_col2:
            st.markdown("#### 🔬 Base-Level Learners vs. Meta-Level Stacking")
            
            # Construct comparison table
            comp_rows = []
            for m_name, probs in base_probas.items():
                row = {"Architecture": "Tier 1: Base Learner", "Model": m_name}
                for i, c_name in enumerate(class_names):
                    row[c_name] = f"{probs[i]*100:.1f}%"
                comp_rows.append(row)

            for m_name, probs in meta_probas.items():
                row = {"Architecture": "Tier 2: Meta Stacking", "Model": m_name}
                for i, c_name in enumerate(class_names):
                    row[c_name] = f"{probs[i]*100:.1f}%"
                comp_rows.append(row)

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Interactive Plotly Radar / Bar agreement visualization
            fig = go.Figure()
            models_list = list(base_probas.keys()) + list(meta_probas.keys())
            for i, c_name in enumerate(class_names):
                vals = [base_probas[b][i] * 100 for b in base_probas] + [meta_probas[m][i] * 100 for m in meta_probas]
                fig.add_trace(go.Bar(
                    name=c_name,
                    x=models_list,
                    y=vals
                ))
            fig.update_layout(
                title="Model Agreement Across Base Learners & Stacking Meta-Models",
                barmode="group",
                xaxis_tickangle=-25,
                height=320,
                margin=dict(l=20, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Clinical Recommendation Action Plan
        st.markdown("---")
        st.markdown("#### 📋 Personalized Clinical Decision Guidance")
        if dataset_key == "brfss":
            c_rec1, c_rec2, c_rec3 = st.columns(3)
            with c_rec1:
                st.info(f"**Glycemic Monitoring**: {'Routine annual check recommended.' if pred_idx == 0 else 'Priority fasting blood glucose and HbA1c screening indicated.'}")
            with c_rec2:
                bmi = input_dict.get("BMI", 25)
                st.warning(f"**Lifestyle Intervention**: {'Maintain balanced diet & activity.' if bmi < 25 else 'Weight reduction target: 5–10% body mass via caloric control.'}")
            with c_rec3:
                bp = input_dict.get("HighBP", 0)
                st.error(f"**Cardiovascular Comorbidity**: {'Normal BP range.' if bp == 0 else 'Active hypertension management required to prevent microvascular damage.'}")
        else:
            r1, r2 = st.columns(2)
            with r1:
                st.info(f"**Discharge Coordination**: {'Standard outpatient care instructions.' if pred_idx == 0 else 'High readmission risk detected: assign dedicated nurse navigator and 7-day post-discharge follow-up.'}")
            with r2:
                st.warning("**Medication Reconciliation**: Verify insulin regimen, oral hypoglycemics, and nephropathy lab panels before release.")

# ======================================================================================
# TAB 2: Model Performance & Comparative Benchmark
# ======================================================================================
with tab_performance:
    st.markdown("### 📊 Held-Out Test Set Evaluation (20% Stratified Split)")
    st.write(
        "Evaluation results measuring the **7 core metrics** highlighted in Section IV-E of Al Reshan et al. (IEEE Access, 2024), "
        "calculated rigorously on the unseen 20% test partition."
    )

    perf_records = []
    # Base DL models
    for name, m_metrics in meta["base_model_results"].items():
        perf_records.append({"Tier": "Tier 1: DL Base", "Model": name, **m_metrics})
    
    # Meta Stacking models
    if "stack_ANN_result" in meta:
        perf_records.append({"Tier": "Tier 2: Meta Stacking", "Model": "Stack-ANN", **meta["stack_ANN_result"]})
    if "stack_LSTM_result" in meta:
        perf_records.append({"Tier": "Tier 2: Meta Stacking", "Model": "Stack-LSTM", **meta["stack_LSTM_result"]})
    if "stack_CNN_result" in meta:
        perf_records.append({"Tier": "Tier 2: Meta Stacking", "Model": "Stack-CNN", **meta["stack_CNN_result"]})
    if "stack_consensus_result" in meta:
        perf_records.append({"Tier": "Tier 2: Meta Stacking", "Model": "Consensus Ensemble", **meta["stack_consensus_result"]})

    perf_df = pd.DataFrame(perf_records)
    
    # Format table
    metric_cols = ["accuracy", "precision", "sensitivity", "specificity", "f_score", "mcc", "roc_auc"]
    styled_df = perf_df.copy()
    for col in metric_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Multi-metric visual chart
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### 📈 Model Accuracy & ROC-AUC Comparison")
        acc_fig = px.bar(
            perf_df,
            x="Model",
            y=["accuracy", "roc_auc"],
            barmode="group",
            color_discrete_sequence=["#00d2be", "#3b82f6"],
            labels={"value": "Score", "variable": "Metric"}
        )
        acc_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(acc_fig, use_container_width=True)

    with col_chart2:
        st.markdown("##### ⚖️ Sensitivity vs. Specificity Tradeoff")
        sens_fig = px.scatter(
            perf_df,
            x="specificity",
            y="sensitivity",
            color="Tier",
            text="Model",
            size=[14] * len(perf_df),
            color_discrete_sequence=["#38bdf8", "#f97316"]
        )
        sens_fig.update_traces(textposition="top center")
        sens_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(sens_fig, use_container_width=True)

    # Extra Trees Classifier (ETC) Feature Importance
    st.markdown("---")
    st.markdown("#### 🌳 Feature Selection (Extra Trees Classifier - Gini Importance Ranking)")
    st.write(
        "Following Step 3 of the Proposed System, an **Extra Trees Classifier (ETC)** ranks feature relevance, "
        "retaining the most salient clinical biomarkers and discarding redundant covariates."
    )
    
    fi_series = pd.Series(meta["feature_importances"]).sort_values(ascending=False).head(15)
    fi_df = pd.DataFrame({"Biomarker / Feature": fi_series.index, "ETC Gini Importance": fi_series.values})
    
    fi_fig = px.bar(
        fi_df,
        x="ETC Gini Importance",
        y="Biomarker / Feature",
        orientation="h",
        color="ETC Gini Importance",
        color_continuous_scale="Viridis"
    )
    fi_fig.update_layout(yaxis=dict(autorange="reversed"), height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fi_fig, use_container_width=True)

# ======================================================================================
# TAB 3: Proposed System Architecture Flow
# ======================================================================================
with tab_flowchart:
    st.markdown("### 🗺️ Proposed System Architecture Flowchart")
    st.write(
        "Interactive architectural blueprint of the **Ensemble Deep Learning (EDL)** framework, "
        "illustrating data transformation from raw health survey records to the final clinical consensus."
    )

    # Architecture diagram rendering
    st.markdown("""
    <div style="background:#0c1524; border:1px solid #1e293b; border-radius:14px; padding:24px; text-align:center;">
        
        <!-- Header -->
        <div style="background:linear-gradient(90deg, #ea580c, #f97316); color:#ffffff; padding:10px 18px; border-radius:8px; font-weight:800; font-size:18px; display:inline-block; margin-bottom:12px;">
            Proposed System Architecture
        </div>
        <div style="color:#94a3b8; font-size:13px; font-style:italic; margin-bottom:20px;">
            Ensemble Deep Learning (EDL) framework — base-level learners + meta-level stacking
        </div>

        <!-- Tier 1 Pipeline Flow -->
        <div style="display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:24px;">
            <div style="background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:12px 18px; color:#f8fafc; font-weight:700; font-size:13px;">
                Diabetes Dataset(s)<br><span style="color:#94a3b8; font-size:11px; font-weight:400;">BRFSS 2015 / Readmit</span>
            </div>
            <div style="color:#38bdf8; font-size:20px;">➔</div>
            <div style="background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:12px 18px; color:#f8fafc; font-weight:700; font-size:13px;">
                Data Preprocessing<br><span style="color:#94a3b8; font-size:11px; font-weight:400;">MinMax (0–1) + Imputation</span>
            </div>
            <div style="color:#38bdf8; font-size:20px;">➔</div>
            <div style="background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:12px 18px; color:#f8fafc; font-weight:700; font-size:13px;">
                Feature Selection (ETC)<br><span style="color:#94a3b8; font-size:11px; font-weight:400;">Extra Trees Classifier (Gini)</span>
            </div>
            <div style="color:#38bdf8; font-size:20px;">➔</div>
            <div style="background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:12px 18px; color:#f8fafc; font-weight:700; font-size:13px;">
                80% Train / 20% Test<br><span style="color:#94a3b8; font-size:11px; font-weight:400;">Stratified Patient Split</span>
            </div>
        </div>

        <div style="color:#f97316; font-size:22px; margin-bottom:14px;">⬇</div>

        <!-- Base Models Box -->
        <div style="background:#0f1c30; border:2px solid #2563eb; border-radius:12px; padding:16px; margin:0 auto 20px auto; max-width:850px;">
            <div style="background:#1d4ed8; color:#ffffff; font-weight:800; padding:4px 12px; border-radius:6px; font-size:12px; display:inline-block; margin-bottom:12px;">
                DL BASE-LEVEL MODELS
            </div>
            <div style="display:flex; justify-content:space-around; flex-wrap:wrap; gap:12px;">
                <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#60a5fa; font-weight:800; font-size:15px;">ANN</div>
                    <div style="color:#94a3b8; font-size:11px; margin-top:4px;">Captures complex non-linear tabular feature interactions</div>
                </div>
                <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#60a5fa; font-weight:800; font-size:15px;">LSTM</div>
                    <div style="color:#94a3b8; font-size:11px; margin-top:4px;">Captures sequential & temporal biomarker health trends</div>
                </div>
                <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#60a5fa; font-weight:800; font-size:15px;">CNN</div>
                    <div style="color:#94a3b8; font-size:11px; margin-top:4px;">Extracts local structural patterns across patient records</div>
                </div>
            </div>
        </div>

        <div style="color:#f97316; font-size:22px; margin-bottom:14px;">⬇ Out-of-Fold Base Probabilities [P_ANN, P_LSTM, P_CNN]</div>

        <!-- Meta Stacking Models Box -->
        <div style="background:#1c1409; border:2px solid #ea580c; border-radius:12px; padding:16px; margin:0 auto 20px auto; max-width:850px;">
            <div style="background:#c2410c; color:#ffffff; font-weight:800; padding:4px 12px; border-radius:6px; font-size:12px; display:inline-block; margin-bottom:12px;">
                EDL META-LEVEL STACKING MODELS
            </div>
            <div style="display:flex; justify-content:space-around; flex-wrap:wrap; gap:12px;">
                <div style="background:#271b12; border:1px solid #f97316; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#fb923c; font-weight:800; font-size:15px;">Stack-ANN</div>
                    <div style="color:#fed7aa; font-size:11px; margin-top:4px;">Dense meta-learner trained on combined probability vectors</div>
                </div>
                <div style="background:#271b12; border:1px solid #f97316; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#fb923c; font-weight:800; font-size:15px;">Stack-LSTM</div>
                    <div style="color:#fed7aa; font-size:11px; margin-top:4px;">Recurrent meta-learner modeling agreement dynamics</div>
                </div>
                <div style="background:#271b12; border:1px solid #f97316; border-radius:8px; padding:12px; flex:1; min-width:200px;">
                    <div style="color:#fb923c; font-weight:800; font-size:15px;">Stack-CNN</div>
                    <div style="color:#fed7aa; font-size:11px; margin-top:4px;">Convolutional meta-learner extracting inter-model correlations</div>
                </div>
            </div>
        </div>

        <div style="color:#00d2be; font-size:22px; margin-bottom:14px;">⬇</div>

        <!-- Final Decision -->
        <div style="background:#032b28; border:2px solid #00d2be; border-radius:12px; padding:14px 28px; display:inline-block;">
            <div style="color:#00d2be; font-weight:800; font-size:16px;">Final Diabetes Risk Prediction</div>
            <div style="color:#a7f3d0; font-size:12px; margin-top:2px;">Calibrated Clinical Risk Probabilities & Class Decision</div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    # Detailed component tabs
    st.markdown("#### 🔍 Architectural Specifications")
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown("""
        **Tier 1: Base Learner Configurations**
        - **ANN Architecture**: `Input(d) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dropout(0.2) -> Output(Softmax/Sigmoid)`
        - **LSTM Architecture**: `Input(d, 1) -> LSTM(32, ReLU) -> Dense(16, ReLU) -> Dropout(0.2) -> Output`
        - **CNN Architecture**: `Input(d, 1) -> Conv1D(32, k=2, ReLU) -> MaxPool1D(2) -> Flatten -> Dense(32) -> Output`
        - **Optimizer**: Adam ($\eta = 0.001$), Loss: Categorical Cross-Entropy / Binary Cross-Entropy
        """)
    with spec_col2:
        st.markdown("""
        **Tier 2: Meta Stacking Configurations**
        - **Input Vector**: $\mathbf{x}_{\text{meta}} = [P_{\text{ANN}}, P_{\text{LSTM}}, P_{\text{CNN}}] \in \mathbb{R}^{3 \times C}$
        - **Stack-ANN**: `Input(meta_d) -> Dense(32, ReLU) -> Dropout(0.2) -> Output`
        - **Stack-LSTM**: `Input(meta_d, 1) -> LSTM(24, ReLU) -> Dense(16, ReLU) -> Output`
        - **Stack-CNN**: `Input(meta_d, 1) -> Conv1D(16, k=2, ReLU) -> MaxPool1D(2) -> Flatten -> Dense(16) -> Output`
        - **Ensemble Consensus**: $\bar{P} = \frac{1}{3} (P_{\text{Stack-ANN}} + P_{\text{Stack-LSTM}} + P_{\text{Stack-CNN}})$
        """)

# ======================================================================================
# TAB 4: Batch Patient CSV Screening
# ======================================================================================
with tab_batch:
    st.markdown("### 📁 Multi-Patient Cohort Batch Screening")
    st.write(
        "Upload a batch CSV file containing multiple patient records. The EDL framework will process all entries "
        "and produce individual model inferences and consensus predictions for download."
    )

    # Download sample template
    col_dl1, col_dl2 = st.columns([1, 2])
    with col_dl1:
        if dataset_key == "brfss":
            sample_df = pd.DataFrame([
                {"BMI": 22.0, "Age": 4, "Income": 7, "PhysHlth": 0, "Education": 5, "MentHlth": 0, "GenHlth": 1, "HighBP": 0, "Fruits": 1, "Smoker": 0, "HighChol": 0, "Veggies": 1, "DiffWalk": 0, "Sex": 0, "PhysActivity": 1},
                {"BMI": 33.5, "Age": 9, "Income": 4, "PhysHlth": 8, "Education": 4, "MentHlth": 5, "GenHlth": 4, "HighBP": 1, "Fruits": 0, "Smoker": 1, "HighChol": 1, "Veggies": 0, "DiffWalk": 1, "Sex": 1, "PhysActivity": 0},
                {"BMI": 28.0, "Age": 7, "Income": 6, "PhysHlth": 2, "Education": 5, "MentHlth": 1, "GenHlth": 3, "HighBP": 1, "Fruits": 1, "Smoker": 0, "HighChol": 0, "Veggies": 1, "DiffWalk": 0, "Sex": 0, "PhysActivity": 1},
            ])
        else:
            sample_df = pd.DataFrame([
                {"time_in_hospital": 3, "num_lab_procedures": 40, "num_medications": 12, "num_procedures": 0, "number_diagnoses": 5, "number_inpatient": 0, "number_emergency": 0, "number_outpatient": 0, "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7},
                {"time_in_hospital": 7, "num_lab_procedures": 65, "num_medications": 24, "num_procedures": 2, "number_diagnoses": 9, "number_inpatient": 2, "number_emergency": 1, "number_outpatient": 0, "discharge_disposition_id": 3, "admission_type_id": 2, "admission_source_id": 7},
            ])
        
        csv_sample = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=csv_sample,
            file_name=f"sample_{dataset_key}_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        batch_raw = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_raw)} patient records**.")
        st.dataframe(batch_raw.head(4), use_container_width=True)

        if st.button("⚡ Execute Batch EDL Inferences", type="primary"):
            with st.spinner("Executing EDL inference across patient cohort..."):
                results_list = []
                for idx, row in batch_raw.iterrows():
                    row_dict = row.to_dict()
                    p_idx, chosen_p, b_p, m_p = predict_edl(
                        scaler, meta, base_models, meta_models, row_dict, selected_meta="Consensus Ensemble"
                    )
                    res_row = {
                        "Patient_ID": f"PT-{idx+1:04d}",
                        "Predicted_Diagnosis": class_names[p_idx],
                        "Risk_Confidence": f"{chosen_p[p_idx]*100:.1f}%",
                        "Stack_ANN_Pred": class_names[int(np.argmax(m_p["Stack-ANN"]))],
                        "Stack_LSTM_Pred": class_names[int(np.argmax(m_p["Stack-LSTM"]))],
                        "Stack_CNN_Pred": class_names[int(np.argmax(m_p["Stack-CNN"]))],
                    }
                    for ci, c_name in enumerate(class_names):
                        res_row[f"Prob_{c_name}"] = f"{chosen_p[ci]:.3f}"
                    results_list.append(res_row)

                batch_res_df = pd.DataFrame(results_list)
                st.success(f"Successfully processed {len(batch_res_df)} patients!")
                st.dataframe(batch_res_df, use_container_width=True)

                res_csv = batch_res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Export Predictions CSV",
                    data=res_csv,
                    file_name=f"edl_{dataset_key}_predictions.csv",
                    mime="text/csv"
                )

# ======================================================================================
# TAB 5: Methodology & Equations
# ======================================================================================
with tab_about:
    st.markdown("### 📖 Scientific Reference & Mathematical Formulation")
    st.markdown("""
    #### 📚 Primary Reference
    > **Al Reshan, M. S., et al.** (2024). *"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"*, **IEEE Access**, vol. 12, pp. 1–15.
    
    ---

    #### 📐 Mathematical Evaluation Formulations (Section IV-E of the Paper)
    
    1. **Sensitivity (Recall / True Positive Rate)**:
       $$\\text{Sensitivity} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$
       
    2. **Specificity (True Negative Rate)**:
       $$\\text{Specificity} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}$$

    3. **Precision (Positive Predictive Value)**:
       $$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$

    4. **F-Score ($F_1$-Measure)**:
       $$F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Sensitivity}}{\\text{Precision} + \\text{Sensitivity}}$$

    5. **Matthews Correlation Coefficient (MCC)**:
       $$\\text{MCC} = \\frac{(\\text{TP} \\times \\text{TN}) - (\\text{FP} \\times \\text{FN})}{\\sqrt{(\\text{TP} + \\text{FP})(\\text{TP} + \\text{FN})(\\text{TN} + \\text{FP})(\\text{TN} + \\text{FN})}}$$

    6. **Meta-Level Stacking Probability Concatenation**:
       $$\\mathbf{z}_i = \\left[ \\hat{P}_{\\text{ANN}}(y|\\mathbf{x}_i) \\,\\Vert\\, \\hat{P}_{\\text{LSTM}}(y|\\mathbf{x}_i) \\,\\Vert\\, \\hat{P}_{\\text{CNN}}(y|\\mathbf{x}_i) \\right]$$
       $$\\hat{y}_{\\text{final}} = \\arg\\max_k \\, \\mathcal{M}_{\\text{stack}}(\\mathbf{z}_i)_k$$

    ---
    
    #### 🏥 Clinical Significance
    Diabetes mellitus and its acute comorbidities require nuanced screening tools capable of synthesizing diverse 
    clinical indicators. By ensembling a feedforward network (ANN), a recurrent network (LSTM), and a convolutional 
    network (CNN), the system captures **tabular interactions**, **sequential patterns**, and **local correlations** 
    simultaneously, outperforming single isolated architectures.
    """)

# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:12px;'>"
    "Diabetes Ensemble Deep Learning (EDL) Clinical Decision Support System • IEEE Access 2024 Replicate Framework"
    "</div>",
    unsafe_allow_html=True
)
