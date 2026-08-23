"""
Insurance Claim Amount Prediction — Streamlit App
Loads the artifacts produced by Insurance_Claim_Prediction_Colab.ipynb:
    - claim_model.pkl
    - gender_encoder.pkl
    - diabetic_encoder.pkl
    - smoker_encoder.pkl
    - region_encoder.pkl
    - feature_columns.pkl

Built by Mirza Yasir Abdullah Baig
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Claim Compass — Insurance Claim Prediction",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR_NAME = "Mirza Yasir Abdullah Baig"
GITHUB_URL = "https://github.com/mirzayasirabdullahbaig07"
LINKEDIN_URL = "https://www.linkedin.com/in/mirza-yasir-abdullah-baig/"
KAGGLE_URL = "https://www.kaggle.com/myab07"

# a reasonable ceiling for the dial - claims rarely run far past this in the dataset
GAUGE_MAX = 55000

# ---------------------------------------------------------------------------
# Global styling — "insurance dashboard / trust" theme
# Built on top of Streamlit's own theme variables (--background-color,
# --secondary-background-color, --text-color) so the app looks right in
# BOTH light and dark mode, whichever the user has selected in Settings.
# Accent colors (sky/coral/green/amber) stay fixed since they read fine
# against either a light or dark surface; everything else derives from
# the active theme via color-mix().
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=Roboto+Mono:wght@400;500;600&display=swap');

:root {
    --sky: #38BDF8;
    --sky-dim: rgba(56, 189, 248, 0.15);
    --coral: #FB7185;
    --green: #2DD4A7;
    --amber: #FBBF24;

    /* Theme-derived tokens — adapt automatically to light/dark */
    --text: var(--text-color);
    --muted: color-mix(in srgb, var(--text-color) 55%, var(--background-color) 45%);
    --surface-solid: var(--secondary-background-color);
    --surface: color-mix(in srgb, var(--secondary-background-color) 72%, transparent);
    --surface-2: color-mix(in srgb, var(--secondary-background-color) 90%, var(--text-color) 10%);
    --border: color-mix(in srgb, var(--text-color) 18%, transparent);
    --border-soft: color-mix(in srgb, var(--text-color) 12%, transparent);
    --sidebar-bg: color-mix(in srgb, var(--background-color) 90%, var(--text-color) 10%);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% -10%, color-mix(in srgb, var(--sky) 7%, transparent) 0%, transparent 40%),
        radial-gradient(circle at 92% 6%, color-mix(in srgb, var(--green) 5%, transparent) 0%, transparent 35%),
        var(--background-color) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding-top: 2rem; }

[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; letter-spacing: 0.1px; }

/* ---------------- Hero header with compass mark ---------------- */
.hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 6px 24px 22px 24px;
    margin: -6px -24px 30px -24px;
    border-bottom: 1px solid var(--border-soft);
    position: relative;
}
.compass {
    position: relative;
    width: 48px; height: 48px;
    border-radius: 50%;
    border: 1px solid var(--sky);
    flex-shrink: 0;
    background: var(--surface-2);
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.2), inset 0 0 10px rgba(56,189,248,0.06);
    display: flex; align-items: center; justify-content: center;
}
.compass-needle {
    width: 2px; height: 32px;
    background: linear-gradient(180deg, var(--coral) 0%, var(--coral) 48%, var(--sky) 52%, var(--sky) 100%);
    border-radius: 2px;
    animation: swing 3.6s ease-in-out infinite;
    transform-origin: center;
}
@keyframes swing { 0%,100% { transform: rotate(-18deg); } 50% { transform: rotate(18deg); } }

.hero-title {
    font-size: 29px; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, var(--text) 25%, var(--sky) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: 'Roboto Mono', monospace;
    font-size: 12px; color: var(--muted); letter-spacing: 1.6px;
    margin: 5px 0 0 0; text-transform: uppercase;
}
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    margin-right: 8px; animation: dotpulse 2s infinite;
}
@keyframes dotpulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ---------------- Glass cards ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 4px 26px rgba(56, 189, 248, 0.08);
}

/* ---------------- Metrics ---------------- */
div[data-testid="stMetric"] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Roboto Mono', monospace !important;
    color: var(--text) !important;
}

/* ---------------- Buttons ---------------- */
.stButton > button {
    background: var(--surface-2) !important;
    color: var(--sky) !important;
    border: 1px solid var(--sky) !important;
    border-radius: 10px !important;
    font-family: 'Roboto Mono', monospace !important;
    letter-spacing: 0.6px;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--sky) !important;
    color: #041017 !important;
    box-shadow: 0 0 22px rgba(56, 189, 248, 0.4);
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stRadio > label { font-family: 'Roboto Mono', monospace; }

/* ---------------- Speedometer gauge ---------------- */
.speedo-wrap {
    display: flex; flex-direction: column; align-items: center;
    padding: 6px 0 4px 0; animation: fadein 0.5s ease;
    color: var(--text); /* lets the inline SVG use currentColor and adapt to theme */
}
@keyframes fadein { from { opacity: 0; transform: scale(0.94); } to { opacity: 1; transform: scale(1); } }
.speedo-value {
    font-family: 'Roboto Mono', monospace; font-size: 34px; font-weight: 700;
    color: var(--text); margin-top: 8px;
}
.speedo-label {
    font-family: 'Roboto Mono', monospace; font-size: 10px;
    letter-spacing: 2px; color: var(--muted); margin-top: 2px;
}
.speedo-band {
    font-family: 'Roboto Mono', monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 999px; margin-top: 10px;
}
.band-low { background: rgba(45, 212, 167, 0.14); color: #0F9C77; border: 1px solid rgba(45,212,167,0.5); }
.band-mid { background: rgba(251, 191, 36, 0.16); color: #B9790B; border: 1px solid rgba(251,191,36,0.5); }
.band-high { background: rgba(251, 113, 133, 0.14); color: #E11D48; border: 1px solid rgba(251,113,133,0.5); }

/* ---------------- Eyebrow labels ---------------- */
.eyebrow {
    font-family: 'Roboto Mono', monospace; font-size: 11px;
    letter-spacing: 2px; text-transform: uppercase; color: var(--sky);
    margin-bottom: 4px;
}

/* ---------------- Sidebar profile ---------------- */
.profile-card {
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 16px 14px;
    background: color-mix(in srgb, var(--sky) 6%, var(--surface-solid) 94%);
    margin-top: 6px;
}
.profile-avatar {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, var(--sky), #0369A1);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 15px;
    color: #041017; flex-shrink: 0;
}
.profile-name {
    font-family: 'Sora', sans-serif; font-weight: 700;
    font-size: 13.5px; color: var(--text); line-height: 1.25; margin: 0;
}
.profile-role {
    font-family: 'Roboto Mono', monospace; font-size: 10px;
    color: var(--muted); letter-spacing: 0.8px; margin: 2px 0 0 0;
}
.profile-links { display: flex; gap: 8px; margin-top: 12px; }
.profile-link {
    flex: 1; text-align: center; text-decoration: none !important;
    font-family: 'Roboto Mono', monospace; font-size: 10.5px; font-weight: 700;
    letter-spacing: 0.5px; color: var(--muted) !important;
    border: 1px solid var(--border-soft); border-radius: 8px;
    padding: 7px 4px; transition: all 0.2s ease;
}
.profile-link:hover {
    color: var(--sky) !important; border-color: var(--sky);
    background: var(--sky-dim); transform: translateY(-1px);
}

/* ---------------- Footer ---------------- */
.app-footer {
    margin-top: 48px; padding: 22px 4px 6px 4px;
    border-top: 1px solid var(--border-soft);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
}
.app-footer .left {
    font-family: 'Roboto Mono', monospace; font-size: 11px; color: var(--muted);
    letter-spacing: 0.4px;
}
.app-footer .left b { color: var(--text); }
.app-footer .links { display: flex; gap: 14px; }
.app-footer .links a {
    font-family: 'Roboto Mono', monospace; font-size: 11px;
    color: var(--muted); text-decoration: none; letter-spacing: 0.4px;
    border-bottom: 1px solid transparent; padding-bottom: 2px;
    transition: all 0.2s ease;
}
.app-footer .links a:hover { color: var(--sky); border-color: var(--sky); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("claim_model.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    diabetic_encoder = joblib.load("diabetic_encoder.pkl")
    smoker_encoder = joblib.load("smoker_encoder.pkl")
    region_encoder = joblib.load("region_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, gender_encoder, diabetic_encoder, smoker_encoder, region_encoder, feature_columns


try:
    (model, gender_encoder, diabetic_encoder, smoker_encoder,
     region_encoder, feature_columns) = load_artifacts()
    ARTIFACTS_OK = True
except FileNotFoundError:
    ARTIFACTS_OK = False


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = gender_encoder.transform(df["gender"])
    df["diabetic"] = diabetic_encoder.transform(df["diabetic"])
    df["smoker"] = smoker_encoder.transform(df["smoker"])
    df["region"] = region_encoder.transform(df["region"])
    return df[feature_columns]


def predict(df_raw: pd.DataFrame):
    X = preprocess(df_raw)
    return model.predict(X)


def claim_band(amount: float):
    if amount >= 25000:
        return "band-high", "HIGH COST"
    elif amount >= 10000:
        return "band-mid", "MODERATE COST"
    return "band-low", "LOW COST"


def render_speedometer(amount: float):
    """Renders a semicircular gauge. Track and needle use currentColor so
    they inherit --text from the wrapping .speedo-wrap div and automatically
    read correctly whether the app is in light or dark mode."""
    pct = min(amount / GAUGE_MAX, 1.0)
    band_class, band_text = claim_band(amount)
    needle_x = 100 + 68 * np.cos(np.pi - np.pi * pct)
    needle_y = 100 - 68 * np.sin(np.pi - np.pi * pct)

    svg = f"""
    <svg viewBox="0 0 200 110" width="220" height="125">
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="currentColor"
            stroke-opacity="0.14" stroke-width="16" stroke-linecap="round"/>
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="url(#grad)" stroke-width="16"
            stroke-linecap="round"
            stroke-dasharray="282.6" stroke-dashoffset="{282.6 * (1 - pct)}"/>
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#2DD4A7"/>
          <stop offset="55%" stop-color="#FBBF24"/>
          <stop offset="100%" stop-color="#FB7185"/>
        </linearGradient>
      </defs>
      <line x1="100" y1="100" x2="{needle_x}" y2="{needle_y}"
            stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="6" fill="currentColor"/>
    </svg>
    """
    st.markdown(f"""
    <div class="speedo-wrap">
      {svg}
      <div class="speedo-value">${amount:,.0f}</div>
      <div class="speedo-label">PREDICTED CLAIM AMOUNT</div>
      <div class="speedo-band {band_class}">{band_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
      <div class="compass"><div class="compass-needle"></div></div>
      <div>
        <p class="hero-title">Claim Compass</p>
        <p class="hero-sub"><span class="status-dot"></span>insurance claim amount prediction system</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(f"""
    <div class="app-footer">
        <div class="left">Claim Compass · built by <b>{AUTHOR_NAME}</b> · scikit-learn + Streamlit</div>
        <div class="links">
            <a href="{GITHUB_URL}" target="_blank">GitHub ↗</a>
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn ↗</a>
            <a href="{KAGGLE_URL}" target="_blank">Kaggle ↗</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0 18px 0;">
        <div class="compass" style="width:34px;height:34px;"><div class="compass-needle" style="height:22px;"></div></div>
        <span style="font-family:'Sora';font-weight:800;font-size:17px;">Claim Compass</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["🎯  Single Estimate", "📊  Batch Prediction", "ℹ️  About"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.caption("Model artifacts loaded from disk · Random Forest tuned with GridSearchCV, evaluated on a held-out test set.")

    initials = "".join([w[0] for w in AUTHOR_NAME.split()[:2]]).upper()
    st.markdown(f"""
    <div class="profile-card">
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="profile-avatar">{initials}</div>
            <div>
                <p class="profile-name">{AUTHOR_NAME}</p>
                <p class="profile-role">ML / DATA SCIENCE</p>
            </div>
        </div>
        <div class="profile-links">
            <a class="profile-link" href="{GITHUB_URL}" target="_blank">GitHub</a>
            <a class="profile-link" href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
            <a class="profile-link" href="{KAGGLE_URL}" target="_blank">Kaggle</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not ARTIFACTS_OK:
    render_hero()
    st.error(
        "Model artifacts not found. Make sure `claim_model.pkl`, `gender_encoder.pkl`, "
        "`diabetic_encoder.pkl`, `smoker_encoder.pkl`, `region_encoder.pkl`, and "
        "`feature_columns.pkl` are in the same folder as this app (produced by the Colab notebook)."
    )
    st.stop()

GENDER_OPTIONS = list(gender_encoder.classes_)
DIABETIC_OPTIONS = list(diabetic_encoder.classes_)
SMOKER_OPTIONS = list(smoker_encoder.classes_)
REGION_OPTIONS = list(region_encoder.classes_)

# ---------------------------------------------------------------------------
# Page: Single Estimate
# ---------------------------------------------------------------------------
if page.startswith("🎯"):
    render_hero()
    st.markdown("<div class='eyebrow'>Estimate · Single Policyholder</div>", unsafe_allow_html=True)
    st.markdown("#### Estimate a policyholder's claim amount")
    st.caption("Enter the policyholder's profile below to get a live claim estimate.")

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>Policyholder Profile</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
                bloodpressure = st.number_input("Blood pressure", min_value=60, max_value=200, value=92)
                children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
            with c2:
                gender = st.selectbox("Gender", GENDER_OPTIONS)
                diabetic = st.selectbox("Diabetic?", DIABETIC_OPTIONS)
                smoker = st.selectbox("Smoker?", SMOKER_OPTIONS)
                region = st.selectbox("Region", REGION_OPTIONS)

            run = st.button("🧭  Estimate Claim", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("<div class='eyebrow' style='text-align:center;'>Result</div>", unsafe_allow_html=True)
            if run:
                row = {
                    "age": age,
                    "gender": gender,
                    "bmi": bmi,
                    "bloodpressure": bloodpressure,
                    "diabetic": diabetic,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                }
                df_input = pd.DataFrame([row])
                pred_amount = predict(df_input)[0]
                render_speedometer(pred_amount)
                st.caption("Estimate based on the trained regression model — treat as a planning figure, not a guarantee.")
            else:
                st.info("Fill in the policyholder profile and click **Estimate Claim** to see the result here.")

# ---------------------------------------------------------------------------
# Page: Batch Prediction
# ---------------------------------------------------------------------------
elif page.startswith("📊"):
    render_hero()
    st.markdown("<div class='eyebrow'>Estimate · Batch</div>", unsafe_allow_html=True)
    st.markdown("#### Estimate claims for a batch of policyholders")
    st.caption(
        "Upload a CSV with columns `age, gender, bmi, bloodpressure, diabetic, "
        "children, smoker, region` (and optionally `claim` for evaluation)."
    )

    with st.container(border=True):
        uploaded = st.file_uploader("Upload policyholder CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required = ["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker", "region"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            bad_gender = set(df["gender"].unique()) - set(GENDER_OPTIONS)
            bad_diabetic = set(df["diabetic"].unique()) - set(DIABETIC_OPTIONS)
            bad_smoker = set(df["smoker"].unique()) - set(SMOKER_OPTIONS)
            bad_region = set(df["region"].unique()) - set(REGION_OPTIONS)
            bad_any = bad_gender or bad_diabetic or bad_smoker or bad_region

            if bad_any:
                st.error(
                    f"Unrecognized category values — gender: {bad_gender or 'none'}, "
                    f"diabetic: {bad_diabetic or 'none'}, smoker: {bad_smoker or 'none'}, "
                    f"region: {bad_region or 'none'}."
                )
            else:
                has_labels = "claim" in df.columns
                preds = predict(df[required])
                results = df.copy()
                results["Predicted_Claim"] = preds

                c1, c2, c3 = st.columns(3)
                c1.metric("TOTAL POLICYHOLDERS", f"{len(df):,}")
                c2.metric("AVG PREDICTED CLAIM", f"${preds.mean():,.0f}")
                c3.metric("MAX PREDICTED CLAIM", f"${preds.max():,.0f}")

                st.markdown("<div class='eyebrow' style='margin-top:18px;'>Results</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    styled = results.sort_values("Predicted_Claim", ascending=False).style.background_gradient(
                        subset=["Predicted_Claim"], cmap="Blues"
                    )
                    st.dataframe(styled, use_container_width=True, height=380)

                    csv_out = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️  Download results as CSV",
                        data=csv_out,
                        file_name="claim_predictions.csv",
                        mime="text/csv",
                    )

                if has_labels:
                    st.markdown("<div class='eyebrow' style='margin-top:18px;'>Evaluation vs Ground Truth</div>", unsafe_allow_html=True)
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    y_true = df["claim"].values

                    with st.container(border=True):
                        p1, p2, p3 = st.columns(3)
                        p1.metric("RMSE", f"${np.sqrt(mean_squared_error(y_true, preds)):,.0f}")
                        p2.metric("MAE", f"${mean_absolute_error(y_true, preds):,.0f}")
                        p3.metric("R² SCORE", f"{r2_score(y_true, preds):.3f}")

                        # Deliberately kept as a plain light-styled chart (default
                        # matplotlib white background, dark text) rather than a
                        # theme-matched dark chart — a light chart card reads fine
                        # sitting on either a light or dark app background, whereas
                        # a hardcoded dark chart looks broken in light mode.
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.scatter(y_true, preds, alpha=0.5, color="#0284C7", s=18)
                        lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
                        ax.plot(lims, lims, color="#E11D48", linestyle="--", linewidth=1.5)
                        ax.set_xlabel("Actual Claim")
                        ax.set_ylabel("Predicted Claim")
                        ax.set_title("Actual vs Predicted")
                        st.pyplot(fig, use_container_width=False)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    render_hero()
    st.markdown("<div class='eyebrow'>System Info</div>", unsafe_allow_html=True)
    st.markdown("#### About Claim Compass")

    with st.container(border=True):
        st.markdown(f"""
This app estimates a policyholder's medical insurance claim amount using a
regression model trained on age, BMI, blood pressure, smoking/diabetic
status, number of children, and region.

**Pipeline**
- Missing values in `age` filled with the median, `region` filled with the mode
- Categorical features (`gender`, `diabetic`, `smoker`, `region`) encoded with
  `LabelEncoder`, with the fitted encoders saved so the app applies the exact
  same mapping used in training
- Compared Linear Regression, Random Forest, and Gradient Boosting
- Best model tuned via `GridSearchCV`, evaluated on a held-out test set with
  RMSE, MAE, and R²

**How to use**
- **Single Estimate** — enter one policyholder's profile for an instant claim estimate
- **Batch Prediction** — upload a CSV of policyholders and estimate claims for
  all of them at once, plus evaluation metrics if you include the true `claim` values

This estimate is a planning figure based on historical patterns, not a
guarantee or an official underwriting decision.

---

**Built by [{AUTHOR_NAME}]({GITHUB_URL})**
[GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL}) · [Kaggle]({KAGGLE_URL})
        """)

render_footer()
