import gradio as gr
import joblib
import pandas as pd

FEATURES = ["koi_period", "koi_duration", "koi_prad", "koi_depth"]
model = joblib.load("model.joblib")

def predict(koi_period, koi_duration, koi_prad, koi_depth):
    X = pd.DataFrame([[koi_period, koi_duration, koi_prad, koi_depth]], columns=FEATURES)
    y = model.predict(X)[0]
    try:
        p = model.predict_proba(X)[0][1]
        label = "Candidate exoplanet" if int(y) == 1 else "Likely false positive"
        return f"{label} | probability: {p:.2f}"
    except Exception:
        return "Candidate exoplanet" if int(y) == 1 else "Likely false positive"

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=f) for f in FEATURES],
    outputs=gr.Textbox(label="Prediction"),
    title="Exoplanet Transit Classifier",
    description="Enter KOI features to get a quick classification."
)

if __name__ == "__main__":
    demo.launch()