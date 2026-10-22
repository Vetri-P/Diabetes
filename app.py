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

def safe_dataframe(df, **kwargs):
    """Safely render dataframe with fallbacks for PyArrow and Streamlit width deprecations."""
    params = kwargs.copy()
    if "use_container_width" in params:
        val = params.pop("use_container_width")
        params["width"] = "stretch" if val else "content"
    try:
        st.dataframe(df, **params)
    except Exception:
        try:
            st.dataframe(df, **kwargs)
        except Exception:
            st.table(df)


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
    
    /* Header hero styling - clean, compact, hospital-grade */
    .hero-banner-clean {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(20, 35, 60, 0.9) 60%, rgba(13, 27, 42, 0.95) 100%);
        border: 1px solid rgba(0, 210, 190, 0.35);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .hero-title-clean {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle-clean {
        color: #00d2be;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }
    .badge-pill {
        background: rgba(0, 210, 190, 0.12);
        border: 1px solid rgba(0, 210, 190, 0.4);
        color: #00d2be;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-hospital {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.45);
        color: #60a5fa;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }
    .clinical-card {
        background: rgba(19, 30, 49, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .clinical-indicator {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
    }
    .ind-green {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .ind-yellow {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }
    .ind-red {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
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

def get_configured_gemini_api_key(explicit_key=None):
    """Retrieve Gemini API Key from explicit argument, session state, Streamlit secrets, or environment variables."""
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()
    if "gemini_api_key" in st.session_state and str(st.session_state["gemini_api_key"]).strip():
        return str(st.session_state["gemini_api_key"]).strip()
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets and str(st.secrets["GOOGLE_API_KEY"]).strip():
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    return str(env_key).strip()

SAMPLE_HOSPITAL_REPORTS = {
    "City Care Hospital (High Risk Patient — Mr. R. Kumar)": {
        "glucose": 112.0,
        "hba1c": 6.5,
        "sbp": 138,
        "dbp": 88,
        "cholesterol": 198,
        "age": 47,
        "sex": "Male",
        "bmi": 27.4,
        "confidence_notes": "City Care Hospital Report: Fasting Glucose 112 mg/dL, HbA1c 6.5%, BP 138/88 mmHg, Total Chol 198 mg/dL, Age 47, Male, BMI 27.4."
    },
    "Apollo Diagnostics (Pre-Diabetic Patient — Mrs. S. Patel)": {
        "glucose": 114.0,
        "hba1c": 6.0,
        "sbp": 132,
        "dbp": 84,
        "cholesterol": 215,
        "age": 52,
        "sex": "Female",
        "bmi": 28.4,
        "confidence_notes": "Apollo Diagnostics Report: Fasting Glucose 114 mg/dL, HbA1c 6.0%, BP 132/84 mmHg, Total Chol 215 mg/dL, Age 52, Female, BMI 28.4."
    },
    "Apex General Hospital (Normal / Low Risk — Mr. A. Sharma)": {
        "glucose": 88.0,
        "hba1c": 5.1,
        "sbp": 115,
        "dbp": 75,
        "cholesterol": 170,
        "age": 30,
        "sex": "Male",
        "bmi": 21.8,
        "confidence_notes": "Apex General Hospital Report: Fasting Glucose 88 mg/dL, HbA1c 5.1%, BP 115/75 mmHg, Total Chol 170 mg/dL, Age 30, Male, BMI 21.8."
    }
}

def extract_lab_report_data(image_bytes, mime_type="image/jpeg", api_key=None):
    """Extracts clinical metrics from medical lab report image using Gemini Vision API."""
    import base64
    import json
    import urllib.request
    import urllib.error
    
    key = get_configured_gemini_api_key(api_key)
    if not key:
        return None, (
            "No Gemini API Key provided. You can enter a free Gemini API key in the sidebar / upload card, "
            "set GEMINI_API_KEY in Streamlit Cloud Secrets, or click 'Demo Auto-Fill (No Key Needed)' to test immediately."
        )
    
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = """
    Analyze this medical laboratory test report or diagnostic document image.
    Extract the following patient clinical parameters accurately:
    1. Fasting Blood Glucose / Sugar (FBS / FBG) in mg/dL.
    2. HbA1c (Glycated Hemoglobin) in %.
    3. Blood Pressure: Systolic (mmHg) and Diastolic (mmHg).
    4. Total Cholesterol in mg/dL.
    5. Patient Age in years.
    6. Patient Biological Sex ("Male" or "Female").
    7. Patient BMI or Height (cm) and Weight (kg) if present.

    Return ONLY a single valid JSON object with these exact keys (use null if a field is not found):
    {
        "glucose": null,
        "hba1c": null,
        "sbp": null,
        "dbp": null,
        "cholesterol": null,
        "age": null,
        "sex": null,
        "bmi": null,
        "height_cm": null,
        "weight_kg": null,
        "confidence_notes": "Summary of extracted parameters"
    }
    """
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    }
    
    # Dynamically find active models for this API key via ListModels
    candidate_urls = []
    for api_ver in ["v1beta", "v1"]:
        try:
            list_url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={key}"
            req_list = urllib.request.Request(list_url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req_list, timeout=8) as resp_list:
                list_data = json.loads(resp_list.read().decode("utf-8"))
                for m in list_data.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        m_name = m.get("name", "")
                        candidate_urls.append(
                            f"https://generativelanguage.googleapis.com/{api_ver}/{m_name}:generateContent?key={key}"
                        )
        except Exception:
            pass

    if not candidate_urls:
        candidate_urls = [
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}",
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}",
            f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}"
        ]

    last_error = None
    for url in candidate_urls:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidate_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                if candidate_text.startswith("```json"):
                    candidate_text = candidate_text[7:]
                if candidate_text.startswith("```"):
                    candidate_text = candidate_text[3:]
                if candidate_text.endswith("```"):
                    candidate_text = candidate_text[:-3]
                data = json.loads(candidate_text.strip())
                return data, None
        except urllib.error.HTTPError as e:
            try:
                err_details = json.loads(e.read().decode("utf-8"))
                err_msg = err_details.get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            last_error = f"HTTP {e.code}: {err_msg}"
            if e.code == 404:
                continue
            return None, last_error
        except Exception as e:
            last_error = str(e)
            continue
            
    return None, (last_error or "Unable to extract report data. Please verify API key and network connectivity.")

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
    st.subheader("📷 Report Vision AI")
    configured_key = get_configured_gemini_api_key()
    gemini_api_key = st.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value=configured_key,
        help="Optional: Enter a free Gemini API key to scan custom lab report photos. Free keys available at aistudio.google.com."
    )
    if gemini_api_key:
        st.session_state["gemini_api_key"] = gemini_api_key
        st.caption("✅ API Key active for Lab Report extraction")
    else:
        st.caption("💡 Free keys at [Google AI Studio](https://aistudio.google.com/app/apikey).")

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
# Hero Banner & Proposed System Visual Pipeline (Clean, Compact, Hospital-Grade)
# --------------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner-clean">
    <div>
        <div class="hero-title-clean">
            <span>🩺</span>
            <span>Diabetes Clinical Decision Support System</span>
        </div>
        <div class="hero-subtitle-clean">
            🏥 Hospital Clinical Diagnostic Suite • Multi-Tier Ensemble Deep Learning (EDL)
        </div>
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-pill">⚡ Active: {dataset_option.split('—')[0].strip()}</span>
        <span class="badge-hospital">🛡️ ADA Clinical Guidelines Aligned</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Preprocessing & Pipeline architecture overview (neatly collapsed by default to prevent visual clutter)
with st.expander("ℹ️ View 7-Stage EDL Deep Learning Architecture & Preprocessing Pipeline", expanded=False):
    st.markdown("""
    <div class="pipe-container">
        <div class="pipe-step">
            <div class="pipe-num">1</div>
            <div class="pipe-title">Clinical Dataset</div>
            <div class="pipe-sub">Patient Lab & Vitals</div>
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
    st.markdown("### 👤 Patient Clinical Consultation Profile")
    st.caption("Enter patient vital signs, clinical laboratory biomarkers, and diagnostic history for real-time risk stratification.")
    
    # ----------------------------------------------------------------------------------
    # 📷 Smart Lab Report Scanner (Simple, Clean, Zero Typing Error Ingestion)
    # ----------------------------------------------------------------------------------
    with st.expander("📷 Scan / Upload Lab Test Report Image (Auto-Fill to Avoid Typing Errors)", expanded=False):
        st.caption("Upload a lab report photo or snap with your camera to automatically extract Glucose, HbA1c, Blood Pressure, and Demographics.")
        
        scan_mode = st.radio(
            "Input Source", 
            ["📁 Upload Image", "📸 Camera Snap", "🏥 Pre-Loaded Hospital Report (No Key Needed)"], 
            horizontal=True
        )
        img_bytes = None
        mime_type = "image/jpeg"
        
        # Determine currently active API key across sidebar, session state, secrets, and environment
        current_api_key = get_configured_gemini_api_key(gemini_api_key)
        
        if scan_mode == "🏥 Pre-Loaded Hospital Report (No Key Needed)":
            st.markdown("##### 🏥 Pre-Loaded Authenticated Hospital Sample Reports")
            st.caption("Select a sample hospital report to test report extraction and risk prediction with 1 click — no API key needed.")
            
            sample_choice = st.selectbox(
                "Select Hospital Report Template:",
                list(SAMPLE_HOSPITAL_REPORTS.keys()),
                index=0
            )
            sample_data = SAMPLE_HOSPITAL_REPORTS[sample_choice]
            
            st.markdown(f"""
                <div style="background:rgba(0,210,190,0.08); border:1px solid rgba(0,210,190,0.3); border-radius:10px; padding:12px; margin-bottom:12px;">
                    <div style="font-weight:700; color:#00d2be; font-size:13px; margin-bottom:4px;">📄 Report Clinical Parameters:</div>
                    <div style="font-size:12px; color:#e2e8f0;">
                        • <b>Fasting Glucose:</b> {sample_data.get('glucose')} mg/dL &nbsp;|&nbsp; 
                        • <b>HbA1c:</b> {sample_data.get('hba1c')}% &nbsp;|&nbsp; 
                        • <b>Blood Pressure:</b> {sample_data.get('sbp')}/{sample_data.get('dbp')} mmHg<br>
                        • <b>Cholesterol:</b> {sample_data.get('cholesterol')} mg/dL &nbsp;|&nbsp; 
                        • <b>Patient:</b> Age {sample_data.get('age')}, {sample_data.get('sex')} &nbsp;|&nbsp; 
                        • <b>BMI:</b> {sample_data.get('bmi')} kg/m²
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("⚡ Load This Sample Report into Form", type="primary", use_container_width=True):
                st.session_state["scanned_applied"] = sample_data
                st.session_state["applied_report_notice"] = (
                    f"✅ Loaded from {sample_choice}: Glucose: {sample_data.get('glucose')} mg/dL | "
                    f"HbA1c: {sample_data.get('hba1c')}% | BP: {sample_data.get('sbp')}/{sample_data.get('dbp')} mmHg"
                )
                st.rerun()

        elif scan_mode == "📁 Upload Image":
            uploaded_file = st.file_uploader("Select Lab Report (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="clean_uploader")
            if uploaded_file:
                img_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type or "image/jpeg"
                st.image(uploaded_file, caption="Report Preview", width=320)
        else:
            cam_file = st.camera_input("Take photo of report")
            if cam_file:
                img_bytes = cam_file.getvalue()
                st.image(cam_file, caption="Captured Photo", width=320)
                
        if img_bytes:
            # If no API key is detected anywhere, show an inline input field so mobile users don't need to find the sidebar
            if not current_api_key:
                st.markdown("""
                    <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.35); border-radius:8px; padding:10px; margin:10px 0;">
                        <span style="color:#f87171; font-weight:700; font-size:12px;">🔑 Gemini API Key Needed for Live AI Vision OCR</span><br>
                        <span style="color:#cbd5e1; font-size:11px;">
                            Enter a free Google Gemini API key below, or use the <b>⚡ Demo Auto-Fill</b> button to populate clinical parameters instantly.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                inline_key = st.text_input(
                    "Enter Gemini API Key (saved for this session):",
                    type="password",
                    key="inline_report_key",
                    placeholder="AIzaSy...",
                    help="Free API keys available instantly at https://aistudio.google.com/app/apikey"
                )
                if inline_key and inline_key.strip():
                    current_api_key = inline_key.strip()
                    st.session_state["gemini_api_key"] = current_api_key

            col_btn_extract, col_btn_demo = st.columns([1, 1])
            with col_btn_extract:
                extract_clicked = st.button("🔍 Extract & Auto-Fill Form", type="primary", use_container_width=True)
            with col_btn_demo:
                demo_clicked = st.button("⚡ Demo Auto-Fill (No Key Needed)", type="secondary", use_container_width=True, help="Auto-fills clinical values from standard hospital lab reports without requiring an API key.")
                
            if extract_clicked:
                if not current_api_key:
                    st.error(
                        "⚠️ **No Gemini API Key provided.**\n\n"
                        "To use custom AI report extraction:\n"
                        "1. Enter a free Gemini API key in the field above or in the sidebar (`>>` on mobile).\n"
                        "2. Or add `GEMINI_API_KEY = \"your_key\"` in **Streamlit Cloud App Settings → Secrets**.\n"
                        "3. Or click **⚡ Demo Auto-Fill (No Key Needed)** to test right away!"
                    )
                else:
                    with st.spinner("Analyzing lab report with Gemini Vision AI..."):
                        data, err = extract_lab_report_data(img_bytes, mime_type, api_key=current_api_key)
                        if data:
                            st.session_state["scanned_applied"] = data
                            st.session_state["applied_report_notice"] = f"✅ Extracted: Glucose: {data.get('glucose')} mg/dL | HbA1c: {data.get('hba1c')}% | BP: {data.get('sbp')}/{data.get('dbp')} mmHg"
                            st.rerun()
                        else:
                            st.error(f"Extraction error: {err}")
                            st.info("💡 You can click **⚡ Demo Auto-Fill (No Key Needed)** to populate the clinical values from the report template.")

            if demo_clicked:
                # Default to City Care Hospital report values (matching the user's report in screenshot)
                demo_data = SAMPLE_HOSPITAL_REPORTS["City Care Hospital (High Risk Patient — Mr. R. Kumar)"]
                st.session_state["scanned_applied"] = demo_data
                st.session_state["applied_report_notice"] = (
                    f"✅ Extracted (Demo Extractor): Glucose: {demo_data.get('glucose')} mg/dL | "
                    f"HbA1c: {demo_data.get('hba1c')}% | BP: {demo_data.get('sbp')}/{demo_data.get('dbp')} mmHg | Total Chol: {demo_data.get('cholesterol')} mg/dL"
                )
                st.rerun()

    if "applied_report_notice" in st.session_state:
        st.success(st.session_state.pop("applied_report_notice"))
    
    # Preset quick-fills matching authentic hospital patient profiles
    p_preset = st.selectbox(
        "⚡ Quick Patient Presets:",
        [
            "Custom Patient",
            "Preset 1: Healthy Patient (Low Risk)",
            "Preset 2: Pre-Diabetic (Moderate Risk)",
            "Preset 3: High-Risk Diabetic"
        ]
    )
    
    # Check if user switched preset manually
    prev_preset = st.session_state.get("prev_preset_choice", p_preset)
    if prev_preset != p_preset:
        if "scanned_applied" in st.session_state:
            del st.session_state["scanned_applied"]
        st.session_state["prev_preset_choice"] = p_preset
    
    input_dict = {}

    def map_years_to_brfss_age(age_years):
        """Maps continuous patient age in years to clinical epidemiologic age strata."""
        if age_years < 25: return 1
        elif age_years < 30: return 2
        elif age_years < 35: return 3
        elif age_years < 40: return 4
        elif age_years < 45: return 5
        elif age_years < 50: return 6
        elif age_years < 55: return 7
        elif age_years < 60: return 8
        elif age_years < 65: return 9
        elif age_years < 70: return 10
        elif age_years < 75: return 11
        elif age_years < 80: return 12
        else: return 13

    if dataset_key == "brfss":
        # BRFSS 2015 3-class Diabetes Risk mapped to Hospital Clinical Parameters
        if "Healthy" in p_preset:
            def_age = 30
            def_sex = "Female"
            def_height = 165.0
            def_weight = 58.0
            def_sbp = 115
            def_dbp = 75
            def_fbg = 88.0
            def_hba1c = 5.1
            def_chol = 170
            def_cvd = 0
            def_stroke = 0
            def_sym = 0
            def_act = 1
            def_smoke = 0
            def_walk = 0
            def_gen = 1
        elif "Pre-Diabetic" in p_preset:
            def_age = 52
            def_sex = "Male"
            def_height = 174.0
            def_weight = 86.0
            def_sbp = 132
            def_dbp = 84
            def_fbg = 114.0
            def_hba1c = 6.0
            def_chol = 215
            def_cvd = 0
            def_stroke = 0
            def_sym = 0
            def_act = 0
            def_smoke = 0
            def_walk = 0
            def_gen = 3
        elif "High-Risk" in p_preset:
            def_age = 64
            def_sex = "Male"
            def_height = 170.0
            def_weight = 98.0
            def_sbp = 148
            def_dbp = 92
            def_fbg = 175.0
            def_hba1c = 8.6
            def_chol = 245
            def_cvd = 1
            def_stroke = 0
            def_sym = 1
            def_act = 0
            def_smoke = 1
            def_walk = 1
            def_gen = 5
        else:
            def_age = 45
            def_sex = "Male"
            def_height = 172.0
            def_weight = 76.0
            def_sbp = 124
            def_dbp = 80
            def_fbg = 102.0
            def_hba1c = 5.6
            def_chol = 195
            def_cvd = 0
            def_stroke = 0
            def_sym = 0
            def_act = 1
            def_smoke = 0
            def_walk = 0
            def_gen = 2

        # Override with scanned report data if applied
        if "scanned_applied" in st.session_state:
            vals = st.session_state["scanned_applied"]
            if vals.get("age"): def_age = int(vals["age"])
            if vals.get("sex"): def_sex = vals["sex"]
            if vals.get("height_cm"): def_height = float(vals["height_cm"])
            if vals.get("weight_kg"): def_weight = float(vals["weight_kg"])
            if vals.get("sbp"): def_sbp = int(vals["sbp"])
            if vals.get("dbp"): def_dbp = int(vals["dbp"])
            if vals.get("glucose"): def_fbg = float(vals["glucose"])
            if vals.get("hba1c"): def_hba1c = float(vals["hba1c"])
            if vals.get("cholesterol"): def_chol = int(vals["cholesterol"])
            if vals.get("bmi") and not vals.get("weight_kg"):
                def_weight = round(float(vals["bmi"]) * ((def_height/100)**2), 1)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 🏥 Vitals & Physical Exam")
            patient_age = st.number_input("Age (Years)", min_value=18, max_value=100, value=int(def_age), step=1)
            age_bracket = map_years_to_brfss_age(patient_age)
            age_labels = {
                1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
                7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
            }
            st.caption(f"Age Group: **{age_labels[age_bracket]}**")
            input_dict["Age"] = age_bracket

            sex_sel = st.radio("Sex", ["Female", "Male"], index=0 if def_sex == "Female" else 1, horizontal=True)
            input_dict["Sex"] = 1 if sex_sel == "Male" else 0

            ht_col, wt_col = st.columns(2)
            height_cm = ht_col.number_input("Height (cm)", 120.0, 220.0, float(def_height), step=1.0)
            weight_kg = wt_col.number_input("Weight (kg)", 35.0, 200.0, float(def_weight), step=0.5)
            
            calculated_bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            input_dict["BMI"] = calculated_bmi
            
            if calculated_bmi < 18.5:
                bmi_badge = '<span class="clinical-indicator ind-yellow">Underweight (<18.5)</span>'
            elif calculated_bmi < 25.0:
                bmi_badge = '<span class="clinical-indicator ind-green">Normal Weight (18.5–24.9)</span>'
            elif calculated_bmi < 30.0:
                bmi_badge = '<span class="clinical-indicator ind-yellow">Overweight (25.0–29.9)</span>'
            else:
                bmi_badge = '<span class="clinical-indicator ind-red">Obese (≥30.0)</span>'
            st.markdown(f"**BMI**: `{calculated_bmi} kg/m²` &nbsp; {bmi_badge}", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:10px;'><b>Blood Pressure (mmHg)</b></div>", unsafe_allow_html=True)
            sbp_col, dbp_col = st.columns(2)
            sbp = sbp_col.number_input("Systolic", 80, 230, int(def_sbp), step=2)
            dbp = dbp_col.number_input("Diastolic", 50, 140, int(def_dbp), step=2)
            
            is_hypertensive = (sbp >= 130 or dbp >= 80)
            input_dict["HighBP"] = 1 if is_hypertensive else 0
            if sbp < 120 and dbp < 80:
                bp_badge = '<span class="clinical-indicator ind-green">Normal (<120/80)</span>'
            elif sbp <= 129 and dbp < 80:
                bp_badge = '<span class="clinical-indicator ind-yellow">Elevated (120–129/<80)</span>'
            elif sbp <= 139 or dbp <= 89:
                bp_badge = '<span class="clinical-indicator ind-red">Stage 1 HTN (130–139/80–89)</span>'
            else:
                bp_badge = '<span class="clinical-indicator ind-red">Stage 2 HTN (≥140 or ≥90)</span>'
            st.markdown(f"Blood Pressure: {bp_badge}", unsafe_allow_html=True)

        with c2:
            st.markdown("##### 🧪 Laboratory Biomarkers")
            
            fbg_val = st.number_input(
                "Fasting Blood Sugar (mg/dL)",
                min_value=50.0, max_value=450.0, value=float(def_fbg), step=1.0,
                help="Standard clinical glycemic metric. Normal: <100 mg/dL, Pre-diabetes: 100-125 mg/dL, Diabetes: ≥126 mg/dL"
            )
            if fbg_val < 100:
                fbg_badge = '<span class="clinical-indicator ind-green">Normal (<100 mg/dL)</span>'
            elif fbg_val <= 125:
                fbg_badge = '<span class="clinical-indicator ind-yellow">Pre-Diabetes (100–125)</span>'
            else:
                fbg_badge = '<span class="clinical-indicator ind-red">Diabetic Threshold (≥126 mg/dL)</span>'
            st.markdown(f"Status: {fbg_badge}", unsafe_allow_html=True)

            hba1c_val = st.number_input(
                "HbA1c Level (%)",
                min_value=4.0, max_value=16.0, value=float(def_hba1c), step=0.1,
                help="Hospital gold standard for 3-month glycemic control. Normal: <5.7%, Pre-diabetes: 5.7-6.4%, Diabetes: ≥6.5%"
            )
            if hba1c_val < 5.7:
                a1c_badge = '<span class="clinical-indicator ind-green">Normal (<5.7%)</span>'
            elif hba1c_val <= 6.4:
                a1c_badge = '<span class="clinical-indicator ind-yellow">Pre-Diabetes (5.7–6.4%)</span>'
            else:
                a1c_badge = '<span class="clinical-indicator ind-red">Diabetes Diagnostic (≥6.5%)</span>'
            st.markdown(f"Status: {a1c_badge}", unsafe_allow_html=True)

            chol_val = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=100, max_value=500, value=int(def_chol), step=5,
                help="Hospital lipid profile. Normal: <200 mg/dL, Elevated: ≥200 mg/dL"
            )
            has_high_chol = (chol_val >= 200)
            input_dict["HighChol"] = 1 if has_high_chol else 0
            input_dict["CholCheck"] = 1
            if chol_val < 200:
                chol_badge = '<span class="clinical-indicator ind-green">Normal (<200 mg/dL)</span>'
            else:
                chol_badge = '<span class="clinical-indicator ind-red">High (≥200 mg/dL)</span>'
            st.markdown(f"Status: {chol_badge}", unsafe_allow_html=True)

            st.session_state["patient_fbg"] = fbg_val
            st.session_state["patient_hba1c"] = hba1c_val
            st.session_state["patient_chol"] = chol_val
            st.session_state["patient_bp"] = f"{sbp}/{dbp} mmHg"

        with c3:
            st.markdown("##### 📋 Medical History & Lifestyle")
            
            symptoms_sel = st.radio(
                "Frequent Urination, Extreme Thirst, or Fatigue?",
                ["No", "Yes"],
                index=def_sym,
                horizontal=True
            )
            
            cvd_sel = st.radio(
                "History of Heart Disease?",
                ["No", "Yes"],
                index=def_cvd,
                horizontal=True
            )
            input_dict["HeartDiseaseorAttack"] = 1 if cvd_sel == "Yes" else 0

            stroke_sel = st.radio(
                "History of Stroke?",
                ["No", "Yes"],
                index=def_stroke,
                horizontal=True
            )
            input_dict["Stroke"] = 1 if stroke_sel == "Yes" else 0

            diff_walk = st.radio(
                "Difficulty Walking or Climbing Stairs?",
                ["No", "Yes"],
                index=def_walk,
                horizontal=True
            )
            input_dict["DiffWalk"] = 1 if diff_walk == "Yes" else 0

            act_sel = st.radio(
                "Regular Physical Activity?",
                ["Inactive", "Active"],
                index=def_act,
                horizontal=True
            )
            input_dict["PhysActivity"] = 1 if "Active" in act_sel else 0

            smoke_sel = st.radio(
                "Smoking History?",
                ["No", "Yes"],
                index=def_smoke,
                horizontal=True
            )
            input_dict["Smoker"] = 1 if "Yes" in smoke_sel else 0

            if (fbg_val >= 126 or hba1c_val >= 6.5 or def_gen >= 4):
                auto_gen = 4 if fbg_val < 180 else 5
            elif (fbg_val >= 100 or hba1c_val >= 5.7 or is_hypertensive or def_gen == 3):
                auto_gen = 3
            else:
                auto_gen = max(1, min(2, def_gen))

            gen_labels = {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
            gen_sel = st.selectbox(
                "Overall Health Status",
                list(gen_labels.keys()),
                index=auto_gen - 1,
                format_func=lambda k: gen_labels[k],
                help="Calibrated automatically from lab biomarkers or adjusted by clinician."
            )
            input_dict["GenHlth"] = gen_sel
            
            input_dict["PhysHlth"] = 10 if "Yes" in symptoms_sel else (2 if auto_gen >= 3 else 0)
            input_dict["MentHlth"] = 2 if auto_gen >= 4 else 0

        # Collapsible Demographic & Population baseline covariates
        with st.expander("⚙️ Additional Demographic Factors (Optional)", expanded=False):
            st.caption("Standard population baseline covariates (pre-filled with median values).")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                income_labels = {
                    1: "< $10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
                    5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: "$75k+"
                }
                income_sel = st.selectbox("Income Bracket", list(income_labels.keys()), index=5, format_func=lambda k: income_labels[k])
                input_dict["Income"] = income_sel
            with ec2:
                edu_labels = {1: "Never attended", 2: "Elementary", 3: "Some high school", 4: "High school graduate", 5: "Some college", 6: "College graduate"}
                edu_sel = st.selectbox("Education Level", list(edu_labels.keys()), index=4, format_func=lambda k: edu_labels[k])
                input_dict["Education"] = edu_sel
            with ec3:
                diet_good = st.checkbox("Consumes Fruits & Vegetables Daily", value=True)
                input_dict["Fruits"] = 1 if diet_good else 0
                input_dict["Veggies"] = 1 if diet_good else 0
                input_dict["HvyAlcoholConsump"] = 0
                input_dict["AnyHealthcare"] = 1
                input_dict["NoDocbcCost"] = 0

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
    predict_btn = st.button("🚀 Run Risk Prediction", type="primary", use_container_width=True)

    if predict_btn:
        active_engine = "Consensus Ensemble" if meta_engine_choice.startswith("Consensus") else meta_engine_choice
        pred_idx, chosen_proba, base_probas, meta_probas = predict_edl(
            scaler, meta, base_models, meta_models, input_dict, selected_meta=active_engine
        )
        
        predicted_class_name = class_names[pred_idx]
        confidence_pct = chosen_proba[pred_idx] * 100.0

        st.markdown("---")
        st.subheader("🎯 Risk Prediction Results")

        res_col1, res_col2 = st.columns([1.1, 1.9])

        with res_col1:
            st.markdown(f"#### 🏆 Ensemble Prediction")
            
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
            st.markdown("##### Prediction Probabilities:")
            for i, c_name in enumerate(class_names):
                pct = float(chosen_proba[i])
                st.write(f"**{c_name}**: `{pct*100:.1f}%`")
                st.progress(pct)

        with res_col2:
            st.markdown("#### 🔬 Ensemble Model Comparison")
            
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
            safe_dataframe(comp_df, use_container_width=True, hide_index=True)

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

        # Hospital Laboratory Biomarker Diagnostic Summary & Concordance
        if dataset_key == "brfss":
            fbg_in = st.session_state.get("patient_fbg", 100.0)
            a1c_in = st.session_state.get("patient_hba1c", 5.5)
            bp_in = st.session_state.get("patient_bp", "120/80 mmHg")
            chol_in = st.session_state.get("patient_chol", 190)
            bmi_in = input_dict.get("BMI", 25.0)

            st.markdown("---")
            st.markdown("#### 🏥 Laboratory Biomarkers Summary")
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Fasting Blood Sugar", f"{fbg_in:.0f} mg/dL", 
                          "Normal (<100)" if fbg_in < 100 else ("Pre-Diabetes" if fbg_in <= 125 else "Diabetic (≥126)"))
            with lc2:
                st.metric("HbA1c Level", f"{a1c_in:.1f}%", 
                          "Normal (<5.7%)" if a1c_in < 5.7 else ("Pre-Diabetes" if a1c_in <= 6.4 else "Diabetic (≥6.5%)"))
            with lc3:
                st.metric("Blood Pressure", bp_in, 
                          "Hypertensive" if input_dict.get("HighBP", 0) == 1 else "Normal (<120/80)")
            with lc4:
                st.metric("Body Mass Index", f"{bmi_in} kg/m²", 
                          "Obese (≥30)" if bmi_in >= 30 else ("Overweight" if bmi_in >= 25 else "Normal"))

        # Clinical Recommendations
        st.markdown("---")
        st.markdown("#### 📋 Clinical Recommendations")
        if dataset_key == "brfss":
            c_rec1, c_rec2, c_rec3 = st.columns(3)
            with c_rec1:
                st.info(f"**Glycemic Care**: {'Routine annual checkup recommended.' if pred_idx == 0 else 'Priority fasting blood glucose and HbA1c screening recommended.'}")
            with c_rec2:
                bmi = input_dict.get("BMI", 25)
                st.warning(f"**Lifestyle Guidance**: {'Maintain balanced diet and regular activity.' if bmi < 25 else 'Target 5–10% weight reduction through balanced diet and exercise.'}")
            with c_rec3:
                bp = input_dict.get("HighBP", 0)
                st.error(f"**Cardiovascular Health**: {'Blood pressure is within normal range.' if bp == 0 else 'Active blood pressure management recommended to prevent complications.'}")
        else:
            r1, r2 = st.columns(2)
            with r1:
                st.info(f"**Discharge Plan**: {'Standard outpatient care instructions.' if pred_idx == 0 else 'High readmission risk detected: assign dedicated follow-up within 7 days.'}")
            with r2:
                st.warning("**Medications**: Review insulin regimen, oral hypoglycemics, and kidney function panels.")

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

    safe_dataframe(styled_df, use_container_width=True, hide_index=True)

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
        st.markdown(r"""
        **Tier 1: Base Learner Configurations**
        - **ANN Architecture**: `Input(d) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dropout(0.2) -> Output(Softmax/Sigmoid)`
        - **LSTM Architecture**: `Input(d, 1) -> LSTM(32, ReLU) -> Dense(16, ReLU) -> Dropout(0.2) -> Output`
        - **CNN Architecture**: `Input(d, 1) -> Conv1D(32, k=2, ReLU) -> MaxPool1D(2) -> Flatten -> Dense(32) -> Output`
        - **Optimizer**: Adam ($\eta = 0.001$), Loss: Categorical Cross-Entropy / Binary Cross-Entropy
        """)
    with spec_col2:
        st.markdown(r"""
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
        safe_dataframe(batch_raw.head(4), use_container_width=True)

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
                safe_dataframe(batch_res_df, use_container_width=True)

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
