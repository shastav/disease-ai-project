# disease-ai-project
A model that predicts the disease from the symptom description and recommends a specialist(s)


# Symptom Analysis AI

An educational machine learning project that analyses user-provided symptom descriptions and predicts the most likely condition from a predefined set of 24 disease categories.

## Features
- Natural-language symptom input
- TF-IDF text processing
- Naive Bayes classification model
- Suggested medical specialists where mappings are available
- Simple, interactive Gradio interface

## Model Performance
- **Model:** Multinomial Naive Bayes
- **Test accuracy:** 94.37%
- **Test dataset:** 462 records

## Important Disclaimer
This project is for educational and demonstration purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

The reported accuracy was measured on a held-out portion of a small, synthetic dataset. It does not represent real-world clinical accuracy.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Gradio
- TF-IDF

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Project Files
- `app.py` — Gradio application
- `disease_model.pkl` — Trained classification model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `specialist_mapping.csv` — Disease-to-specialist mappings
- `requirements.txt` — Python dependencies
