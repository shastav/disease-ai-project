
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import html

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HealthAI | Symptom Analysis",
    page_icon="✚",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_files():
    model = joblib.load(BASE_DIR / "disease_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    specialists_df = pd.read_csv(BASE_DIR / "specialist_mapping.csv")
    return model, vectorizer, specialists_df

model, vectorizer, specialists_df = load_files()

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #F8F6F0;
    color: #29394B;
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Main content width */
.block-container {
    max-width: 880px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Header */
.title {
    color: #263B53;
    text-align: center;
    font-size: 36px;
    font-weight: 750;
    margin-bottom: 8px;
}

.subtitle {
    color: #718092;
    text-align: center;
    font-size: 15px;
    margin-bottom: 35px;
}

/* Section headings */
h2, h3 {
    color: #263B53 !important;
}

/* Text */
p, label {
    color: #29394B;
}

/* Text input */
.stTextArea textarea {
    background-color: #FCFBF8 !important;
    color: #29394B !important;
    border: 1px solid #E5DED1 !important;
    border-radius: 18px !important;
    padding: 17px !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

.stTextArea textarea::placeholder {
    color: #84909D !important;
}

/* Buttons */
.stButton > button {
    background-color: #263B53 !important;
    color: #FFFFFF !important;
    border: 1px solid #263B53 !important;
    border-radius: 13px !important;
    min-height: 48px;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100%;
}

/* Force button text to remain visible */
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.stButton > button:hover {
    background-color: #1B2D42 !important;
    border-color: #1B2D42 !important;
    color: #FFFFFF !important;
}

/* Result cards */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E7E2D8;
    border-radius: 16px;
    padding: 22px;
    margin-top: 12px;
    margin-bottom: 15px;
}

.result-label {
    color: #718092;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 9px;
}

.result-value {
    color: #263B53;
    font-size: 23px;
    font-weight: 700;
}

/* Specialist text */
.specialist-item {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 5px;
    margin: 12px 0;
}

.specialist-name {
    color: #183153 !important;
    font-size: 15px;
    font-weight: 750;
}

.specialist-detail {
    color: #596777 !important;
    font-size: 14px;
}

/* Disclaimer */
.disclaimer {
    background: #F1EFE9;
    border: 1px solid #E7E2D8;
    border-radius: 14px;
    padding: 17px;
    margin-top: 25px;
    color: #35475A;
    font-size: 13px;
    line-height: 1.8;
}

@media (max-width: 600px) {
    .title {
        font-size: 29px;
    }

    .block-container {
        padding-top: 1.5rem;
    }
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. CLEAR INPUT FUNCTION
# ==========================================

def clear_input():
    st.session_state["symptoms"] = ""

# ==========================================
# 5. HEADER
# ==========================================

st.markdown(
    '<div class="title">HealthAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Symptom exploration and medical specialist information'
    '</div>',
    unsafe_allow_html=True
)

# ==========================================
# 6. SYMPTOM INPUT
# ==========================================

st.markdown("### Describe your symptoms")

st.markdown(
    "Enter a description using your own words. "
    "Include the symptoms you are experiencing."
)

symptoms = st.text_area(
    label="Symptoms",
    placeholder=(
        "For example: I have been experiencing "
        "a headache, fever and body aches..."
    ),
    height=155,
    label_visibility="collapsed",
    key="symptoms"
)

col1, col2 = st.columns([4, 1])

with col1:
    analyse = st.button(
        "Analyse symptoms",
        key="analyse_button",
        type="primary",
        use_container_width=True
    )

with col2:
    st.button(
        "Clear",
        on_click=clear_input,
        use_container_width=True
    )

# ==========================================
# 7. PREDICTION
# ==========================================

if analyse:

    if not symptoms or not symptoms.strip():

        st.warning("Please enter your symptoms first.")

    else:

        symptoms_vec = vectorizer.transform([symptoms])
        prediction = model.predict(symptoms_vec)[0]

        specialists = specialists_df[
            specialists_df["dataset_disease"] == prediction
        ]

        st.markdown("---")
        st.markdown("## Results")

        # Predicted condition
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">PREDICTED CONDITION</div>
                <div class="result-value">
                    {html.escape(str(prediction))}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Specialist information
        st.markdown("### Relevant medical specialists")

        specialist_html = ""

        if len(specialists) > 0:

            for _, row in specialists.iterrows():

                specialist = html.escape(str(row["specialist"]))
                match_type = html.escape(str(row["match_type"]))
                weight = html.escape(str(row["weight"]))

                if match_type == "Manual project routing":
                    detail = "Manual project assignment"
                else:
                    detail = f"{match_type}, {weight}"

                specialist_html += f"""
                <div class="specialist-item">
                    <span class="specialist-name">
                        {specialist}
                    </span>
                    <span class="specialist-detail">
                        — {detail}
                    </span>
                </div>
                """

        else:

            specialist_html = """
            <p class="specialist-detail">
                No specialist information is available.
            </p>
            """

        st.markdown(
            f"""
            <div class="result-card">
                {specialist_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Specialist suggestions are project-generated routing "
            "information, not verified medical referrals."
        )

# ==========================================
# 8. ACCURACY AND DISCLAIMER
# ==========================================

# ==========================================
# 8. ACCURACY AND DISCLAIMER
# ==========================================

# ==========================================
# 8. ACCURACY AND DISCLAIMER
# ==========================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Model performance:</strong><br>
        94.37% accuracy on a held-out test set of 462 records
        from the project dataset.

        <br><br>

        <strong>Important:</strong><br>
        This is an educational AI project, not a medical
        diagnostic tool. The reported accuracy reflects
        performance on this dataset and does not represent
        real-world clinical accuracy. Predictions and specialist
        suggestions may be inaccurate. Consult a qualified
        healthcare professional for medical concerns.
    </div>
    """,
    unsafe_allow_html=True
)
