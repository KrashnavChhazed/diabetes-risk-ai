"""
Healthcare AI — Diabetes Risk Predictor (Streamlit Web App)
============================================================
Loads:
  • large_diabetes_model.pkl   (XGBoost classifier)
  • preprocessor.pkl           (ColumnTransformer: OneHotEncoder for gender/smoking_history,
                                passthrough for age, hypertension, heart_disease, bmi,
                                HbA1c_level, blood_glucose_level, age_glucose)
  • decision_threshold.pkl     (float = 0.4)

Engineered feature:  age_glucose = age × blood_glucose_level

Preprocessor input columns (exact, confirmed from pkl):
  ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
   'bmi', 'HbA1c_level', 'blood_glucose_level', 'age_glucose']

Run:
  streamlit run web_app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_FILE     = "large_diabetes_model.pkl"
PREPROC_FILE   = "preprocessor.pkl"
THRESHOLD_FILE = "decision_threshold.pkl"

GENDER_OPTIONS  = ["Male", "Female"]
YESNO_OPTIONS   = ["No", "Yes"]
SMOKING_OPTIONS = ["never", "former", "current", "ever", "not current", "No Info"]

DISCLAIMER = (
    "⚕️  **Medical Disclaimer:** This tool provides an AI-generated risk estimate "
    "for informational purposes only. It is **not** a medical diagnosis and should "
    "**not** replace professional medical advice, examination, or testing. "
    "Always consult a qualified healthcare provider regarding any health concerns."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def app_dir() -> str:
    """Return the directory containing this script (or the PyInstaller bundle root)."""
    if getattr(sys, "_MEIPASS", None):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Loading model artefacts…")
def load_artifacts():
    """Load and cache ML artefacts from the same directory as this script."""
    base = app_dir()
    missing = [
        f for f in (MODEL_FILE, PREPROC_FILE, THRESHOLD_FILE)
        if not os.path.exists(os.path.join(base, f))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing file(s) — place them next to web_app.py:\n  " +
            "\n  ".join(missing)
        )

    model   = joblib.load(os.path.join(base, MODEL_FILE))
    preproc = joblib.load(os.path.join(base, PREPROC_FILE))
    raw_thr = joblib.load(os.path.join(base, THRESHOLD_FILE))

    if isinstance(raw_thr, dict):
        threshold = float(raw_thr.get("threshold", 0.5))
    elif hasattr(raw_thr, "__len__"):
        threshold = float(raw_thr[0])
    else:
        threshold = float(raw_thr)

    return model, preproc, threshold


def predict(model, preproc, threshold, inputs: dict) -> tuple[str, str, float]:
    """
    Run preprocessing + inference.

    Returns
    -------
    category : str   — human-readable risk label
    color    : str   — Streamlit-compatible colour name
    pct      : float — probability × 100
    """
    age_glucose = inputs["age"] * inputs["blood_glucose_level"]

    X = pd.DataFrame([{
        "gender":              inputs["gender"],
        "age":                 inputs["age"],
        "hypertension":        inputs["hypertension"],
        "heart_disease":       inputs["heart_disease"],
        "smoking_history":     inputs["smoking_history"],
        "bmi":                 inputs["bmi"],
        "HbA1c_level":         inputs["HbA1c_level"],
        "blood_glucose_level": inputs["blood_glucose_level"],
        "age_glucose":         age_glucose,
    }])

    X_proc = preproc.transform(X)
    proba  = float(model.predict_proba(X_proc)[0][1])
    pct    = proba * 100

    if proba >= threshold:
        if pct >= 65:
            category, color = "🔴  High Risk",     "red"
        else:
            category, color = "🟡  Moderate Risk", "orange"
    else:
        category, color = "🟢  Low Risk", "green"

    return category, color, pct


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Do I Have Diabetes? — Healthcare AI",
    page_icon="🩺",
    layout="centered",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── General ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }

    /* ── Headers ── */
    h1 { color: #FFFFFF !important; font-size: 2rem !important; }
    h3 { color: #42A5F5 !important; font-size: 0.85rem !important;
         text-transform: uppercase; letter-spacing: 0.08em; }

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        background-color: #2C2C2C !important;
        color: #FFFFFF !important;
        border-color: #2A2A2A !important;
    }

    /* ── Labels ── */
    label, .stSelectbox label, .stNumberInput label {
        color: #B0BEC5 !important;
    }

    /* ── Divider ── */
    hr { border-color: #2A2A2A !important; }

    /* ── Button ── */
    [data-testid="baseButton-primary"] {
        background-color: #1E88E5 !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 0 !important;
        border-radius: 6px !important;
        width: 100% !important;
        color: #FFFFFF !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background-color: #1565C0 !important;
    }

    /* ── Result card ── */
    .result-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
        border: 1px solid #2A2A2A;
    }
    .result-category {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .result-prob {
        font-size: 0.95rem;
        color: #B0BEC5;
    }

    /* ── Disclaimer box ── */
    [data-testid="stInfo"] {
        background-color: #1A237E22 !important;
        border-left-color: #1E88E5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load artefacts ────────────────────────────────────────────────────────────

model_ok   = True
model_err  = ""
model      = preproc = threshold = None

try:
    model, preproc, threshold = load_artifacts()
except Exception as exc:
    model_ok  = False
    model_err = str(exc)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Do I Have Diabetes?")
st.caption("AI-powered risk estimation from your health indicators")

if model_ok:
    st.success("✓  Model ready", icon=None)
else:
    st.error(f"✗  Model unavailable — {model_err}", icon=None)

st.markdown("---")


# ── Input form ────────────────────────────────────────────────────────────────

with st.container():

    # ── Section: Demographics ──────────────────────────────────────────────
    st.markdown("### 👤 Demographics")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender *",
            options=[""] + GENDER_OPTIONS,
            index=0,
            help="Select biological sex as recorded in your health records.",
        )

    with col2:
        age = st.number_input(
            "Age *",
            min_value=1.0,
            max_value=130.0,
            value=None,
            step=1.0,
            format="%.0f",
            placeholder="e.g. 45",
            help="Your current age in years (1–130).",
        )

    st.markdown("---")

    # ── Section: Medical History ───────────────────────────────────────────
    st.markdown("### 🏥 Medical History")
    col3, col4, col5 = st.columns(3)

    with col3:
        hypert_raw = st.selectbox(
            "Hypertension *",
            options=[""] + YESNO_OPTIONS,
            index=0,
            help="Do you have a diagnosis of high blood pressure?",
        )

    with col4:
        heart_raw = st.selectbox(
            "Heart Disease *",
            options=[""] + YESNO_OPTIONS,
            index=0,
            help="Have you been diagnosed with any heart disease?",
        )

    with col5:
        smoking = st.selectbox(
            "Smoking History *",
            options=[""] + SMOKING_OPTIONS,
            index=0,
            help="Your current or historical smoking status.",
        )

    st.markdown("---")

    # ── Section: Biometrics ────────────────────────────────────────────────
    st.markdown("### 📊 Biometrics")
    col6, col7, col8 = st.columns(3)

    with col6:
        bmi = st.number_input(
            "BMI *",
            min_value=5.0,
            max_value=100.0,
            value=None,
            step=0.1,
            format="%.1f",
            placeholder="e.g. 27.3",
            help="Body Mass Index (5–100).",
        )

    with col7:
        hba1c = st.number_input(
            "HbA1c Level",
            min_value=0.0,
            max_value=20.0,
            value=5.5,
            step=0.1,
            format="%.1f",
            help="Glycated haemoglobin (%) — leave at default (5.5) if unknown.",
        )

    with col8:
        glucose = st.number_input(
            "Blood Glucose Level *",
            min_value=20.0,
            max_value=1000.0,
            value=None,
            step=1.0,
            format="%.0f",
            placeholder="e.g. 120",
            help="Fasting or random blood glucose in mg/dL (20–1000).",
        )

    st.markdown("---")

    # ── Predict button ─────────────────────────────────────────────────────
    run = st.button("Assess My Risk", type="primary", use_container_width=True)


# ── Validation & Prediction ───────────────────────────────────────────────────

if run:
    errors = []

    if not gender:
        errors.append("Gender is required.")
    if age is None:
        errors.append("Age is required.")
    if not hypert_raw:
        errors.append("Hypertension is required.")
    if not heart_raw:
        errors.append("Heart Disease is required.")
    if not smoking:
        errors.append("Smoking History is required.")
    if bmi is None:
        errors.append("BMI is required.")
    if glucose is None:
        errors.append("Blood Glucose Level is required.")

    if errors:
        for err in errors:
            st.error(f"⚠  {err}")
    elif not model_ok:
        st.error(f"⚠  Cannot run prediction — {model_err}")
    else:
        inputs = {
            "gender":              gender,
            "age":                 float(age),
            "hypertension":        1 if hypert_raw == "Yes" else 0,
            "heart_disease":       1 if heart_raw  == "Yes" else 0,
            "smoking_history":     smoking,
            "bmi":                 float(bmi),
            "HbA1c_level":         float(hba1c),
            "blood_glucose_level": float(glucose),
        }

        try:
            category, color, pct = predict(model, preproc, threshold, inputs)

            color_hex = {"red": "#EF5350", "orange": "#FFC107", "green": "#66BB6A"}[color]
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-category" style="color:{color_hex};">{category}</div>
                    <div class="result-prob">
                        Estimated probability: <strong>{pct:.1f}%</strong>
                        &nbsp;·&nbsp; Decision threshold ≥ {threshold:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"⚠  Prediction error: {exc}")


# ── Disclaimer ────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.info(DISCLAIMER)
