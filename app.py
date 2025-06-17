import gradio as gr
import pandas as pd
from lightcurve_utils import extract_features, detect_anomaly, plot_lightcurve, predict_future_flux, plot_prediction


def process_lightcurve(file):
    df = pd.read_csv(file.name)

    # Plot and save lightcurve
    plot_lightcurve(df)

    # Anomaly detection
    label, model = detect_anomaly(df)

    # Future prediction
    future_times, future_flux = predict_future_flux(df)
    plot_prediction(df, future_times, future_flux)

    return (
        label,
        "lightcurve_plot.png",
        "predicted_flux_plot.png"
    )


demo = gr.Interface(
    fn=process_lightcurve,
    inputs=gr.File(label="Upload lightcurve CSV"),
    outputs=[
        gr.Text(label="Anomaly Prediction"),
        gr.Image(type="filepath", label="Lightcurve"),
        gr.Image(type="filepath", label="Predicted Flux"),
    ],
    title="Lightcurve Anomaly Detector & Flux Predictor",
    description="Upload a CSV with columns: time, flux, flux_err"
)

if __name__ == "__main__":
    demo.launch()