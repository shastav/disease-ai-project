
import gradio as gr
import pandas as pd
import joblib
import os

# ==========================================
# 1. LOAD MODEL FILES
# ==========================================

model = joblib.load("disease_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
specialists_df = pd.read_csv("specialist_mapping.csv")


# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================

def predict_disease(symptoms):

    if not symptoms or not symptoms.strip():
        return (
            "Please enter your symptoms first.",
            "Specialist information will appear here."
        )

    symptoms_vec = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vec)[0]

    specialists = specialists_df[
        specialists_df["dataset_disease"] == prediction
    ]

    specialist_output = ""

    if len(specialists) > 0:
        for _, row in specialists.iterrows():

            specialist = row["specialist"]
            match_type = row["match_type"]
            weight = row["weight"]

            if match_type == "Manual project routing":
                specialist_output += (
                    f"- **{specialist}** — Manual project assignment\n\n"
                )
            else:
                specialist_output += (
                    f"- **{specialist}** — {match_type}, {weight}\n\n"
                )
    else:
        specialist_output = "No specialist information is available."

    return prediction, specialist_output


# ==========================================
# 3. CUSTOM CSS
# ==========================================

css = """
/* Main page */
.gradio-container {
    background: #F8F6F0 !important;
    color: #29394B !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* Main card */
#main-card {
    max-width: 880px !important;
    margin: 35px auto !important;
    padding: 36px !important;
    background: #FFFFFF !important;
    border: 1px solid #E7E2D8 !important;
    border-radius: 22px !important;
    box-shadow: 0 8px 28px rgba(38,59,83,0.045) !important;
}

/* Header */
#app-title h1 {
    color: #263B53 !important;
    text-align: center !important;
    font-size: 34px !important;
    font-weight: 700 !important;
}

#app-subtitle p {
    color: #718092 !important;
    text-align: center !important;
    font-size: 15px !important;
}

#input-heading h3 {
    color: #263B53 !important;
    font-size: 18px !important;
}

#instructions p {
    color: #718092 !important;
    font-size: 14px !important;
}

/* Textbox shell */
#input-shell {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Remove default Gradio textbox backgrounds */
#input-shell,
#input-shell > div,
#input-shell [data-testid="textbox"],
#input-shell .wrap,
#input-shell label {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Actual textarea */
#input-shell textarea,
textarea#symptoms-input {
    background: #FCFBF8 !important;
    background-color: #FCFBF8 !important;
    color: #29394B !important;
    -webkit-text-fill-color: #29394B !important;
    caret-color: #263B53 !important;
    border: 1px solid #E5DED1 !important;
    border-radius: 20px !important;
    padding: 20px !important;
    min-height: 155px !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Placeholder */
#input-shell textarea::placeholder,
textarea#symptoms-input::placeholder {
    color: #84909D !important;
    -webkit-text-fill-color: #84909D !important;
    opacity: 1 !important;
}

/* Focus */
#input-shell textarea:focus,
textarea#symptoms-input:focus {
    border: 1px solid #71869C !important;
    box-shadow: 0 0 0 3px rgba(38,59,83,0.08) !important;
}

/* Analyse button */
#analyse-btn button {
    background: #263B53 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #263B53 !important;
    border-radius: 13px !important;
    min-height: 51px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

#analyse-btn button *,
#analyse-btn button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

#analyse-btn button:hover {
    background: #1B2D42 !important;
}

/* Clear button */
#clear-btn button {
    background: #F3F0E9 !important;
    color: #263B53 !important;
    -webkit-text-fill-color: #263B53 !important;
    border: 1px solid #E7E2D8 !important;
    border-radius: 13px !important;
    min-height: 51px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

/* Results */
#results-heading h2 {
    color: #263B53 !important;
    font-size: 22px !important;
}

#result-card,
#specialist-card {
    background: #FFFFFF !important;
    border: 1px solid #E7E2D8 !important;
    border-radius: 16px !important;
    padding: 19px !important;
    box-shadow: none !important;
}

#result-card h3,
#specialist-card h3 {
    color: #263B53 !important;
}

#result-card p,
#specialist-card p,
#result-card li,
#specialist-card li {
    color: #29394B !important;
    line-height: 1.8 !important;
}

#disclaimer, #disclaimer * {
    color: #35475A !important;
    -webkit-text-fill-color: #35475A !important;
}
}

#disclaimer p {
    color: #596777 !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
}

/* Mobile */
@media (max-width: 600px) {
    #main-card {
        margin: 10px auto !important;
        padding: 19px !important;
    }

    #app-title h1 {
        font-size: 29px !important;
    }

    #specialist-output strong {
    color: #244B75 !important;
    -webkit-text-fill-color: #244B75 !important;
}
/* Specialist names */
#specialist-output strong,
#specialist-output strong *,
#specialist-output b,
#specialist-output b * {
    color: #183153 !important;
    -webkit-text-fill-color: #183153 !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

/* All specialist output text */
#specialist-output,
#specialist-output p,
#specialist-output li {
    color: #29394B !important;
    -webkit-text-fill-color: #29394B !important;
    opacity: 1 !important;
}
#predict-button,
#predict-button button {
    background: #244B75 !important;
    background-color: #244B75 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
}

#predict-button:hover,
#predict-button button:hover {
    background: #183653 !important;
    background-color: #183653 !important;
}
}
"""


# ==========================================
# 4. BUILD INTERFACE
# ==========================================

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=css,
    title="Symptom Analysis"
) as demo:

    with gr.Column(elem_id="main-card"):

        gr.Markdown(
            "# HealthAI",
            elem_id="app-title"
        )

        gr.Markdown(
            "Symptom exploration and medical specialist information",
            elem_id="app-subtitle"
        )

        gr.Markdown(
            "### Describe your symptoms",
            elem_id="input-heading"
        )

        gr.Markdown(
            "Enter a description using your own words. "
            "Include the symptoms you are experiencing.",
            elem_id="instructions"
        )

        # Textbox without the default Gradio container
        with gr.Column(elem_id="input-shell"):
            symptoms_input = gr.Textbox(
                show_label=False,
                container=False,
                placeholder=(
                    "For example: I have been experiencing "
                    "a headache, fever and body aches..."
                ),
                lines=5,
                elem_id="symptoms-input"
            )

        # Buttons
        with gr.Row():

            analyse_btn = gr.Button(
                "Analyse symptoms",
                variant="primary",
                elem_id="analyse-btn",
                scale=4
            )

            clear_btn = gr.Button(
                "Clear",
                variant="secondary",
                elem_id="clear-btn",
                scale=1
            )

        # Results
        gr.Markdown(
            "## Results",
            elem_id="results-heading"
        )

        with gr.Column(elem_id="result-card"):
            gr.Markdown("### Predicted condition")

            disease_output = gr.Markdown(
                "Your result will appear here."
            )

        with gr.Column(elem_id="specialist-card"):
            gr.Markdown("### Relevant medical specialists")

            specialist_output = gr.Markdown(
                "Specialist information will appear here."
            )

        # Accuracy and disclaimer
        gr.Markdown(
            """
            **Model performance:** 94.37% accuracy on a held-out "
            test set of 462 records from the project dataset.

            **Important:** This is an educational AI project, not a "
            medical diagnostic tool. The reported accuracy reflects "
            performance on this dataset and does not represent "
            real-world clinical accuracy. Predictions and specialist "
            suggestions may be inaccurate. Consult a qualified "
            healthcare professional for medical concerns.
            """,
            elem_id="disclaimer"
        )

    # Button actions
    analyse_btn.click(
        fn=predict_disease,
        inputs=symptoms_input,
        outputs=[disease_output, specialist_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[
            symptoms_input,
            disease_output,
            specialist_output
        ]
    )


# ==========================================
# 5. LAUNCH
# ==========================================

import socket

def find_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=css
    )
