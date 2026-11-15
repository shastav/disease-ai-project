
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

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

/* Main content */
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

/* Headings */
h2, h3 {
    color: #263B53 !important;
}

/* General text */
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

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.stButton > button:hover {
    background-color: #1B2D42 !important;
    border-color: #1B2D42 !important;
}

/* Result card */
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

/* Specialist names */
.specialist-heading {
    color: #183153;
    font-size: 16px;
    font-weight: 700;
}

/* Disclaimer */
.disclaimer-text {
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

        try:
            symptoms_vec = vectorizer.transform([symptoms])
            prediction = model.predict(symptoms_vec)[0]

            # Find the correct disease column
            if "dataset_disease" in specialists_df.columns:
                disease_column = "dataset_disease"
            elif "disease" in specialists_df.columns:
                disease_column = "disease"
            else:
                st.error(
                    "The specialist mapping file is missing "
                    "its disease column."
                )
                st.stop()

            if "specialist" not in specialists_df.columns:
                st.error(
                    "The specialist mapping file is missing "
                    "its specialist column."
                )
                st.stop()

            # Case-insensitive disease matching
            specialists = specialists_df[
                specialists_df[disease_column]
                .astype(str)
                .str.strip()
                .str.casefold()
                == str(prediction).strip().casefold()
            ].copy()

            st.markdown("---")
            st.markdown("## Results")

            # Predicted condition
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTED CONDITION
                    </div>
                    <div class="result-value">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ======================================
            # SPECIALIST RECOMMENDATIONS
            # ======================================

            st.markdown("### Relevant medical specialists")

            if not specialists.empty:

                # Remove duplicate specialist names
                specialists = specialists.drop_duplicates(
                    subset=["specialist"]
                )

                for _, row in specialists.iterrows():

                    specialist = str(row["specialist"]).strip()

                    if not specialist or specialist.lower() == "nan":
                        continue

                    match_type = str(
                        row.get("match_type", "")
                    ).strip()

                    weight = str(
                        row.get("weight", "")
                    ).strip()

                    # Create readable descriptions
                    if match_type == "Manual project routing":

                        detail = "Manual project assignment"

                    elif match_type and match_type.lower() != "nan":

                        if weight and weight.lower() != "nan":
                            detail = f"{match_type} · {weight}"
                        else:
                            detail = match_type

                    else:

                        detail = "Project mapping"

                    # Native Streamlit rendering, not HTML
                    with st.container(border=True):

                        st.markdown(
                            f"**{specialist}**"
                        )

                        st.caption(detail)

            else:

                with st.container(border=True):
                    st.write(
                        "No specialist information is available "
                        "for this predicted condition."
                    )

            st.caption(
                "Specialist suggestions are project-generated "
                "routing information, not verified medical referrals."
            )

        except Exception as e:

            st.error(
                "Something went wrong while analysing the input."
            )

            st.exception(e)

# ==========================================
# 8. ACCURACY AND DISCLAIMER
# ==========================================

st.markdown("---")

with st.container(border=True):

    st.markdown("**Model performance**")

    st.markdown(
        '<div class="disclaimer-text">'
        '94.37% accuracy on a held-out test set of 462 records '
        'from the project dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("")

    st.markdown("**Important**")

    st.markdown(
        """
        This is an educational AI project, not a medical
        diagnostic tool.

        The reported accuracy reflects performance on this
        dataset and does not represent real-world clinical accuracy.

        Predictions and specialist suggestions may be inaccurate.
        Consult a qualified healthcare professional for medical
        concerns.
        """
    )
