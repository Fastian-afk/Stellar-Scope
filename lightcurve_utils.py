import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

def extract_features(df):
    flux = df['flux'].values
    features = {
        "mean_flux": np.mean(flux),
        "std_flux": np.std(flux),
        "min_flux": np.min(flux),
        "max_flux": np.max(flux),
        "median_flux": np.median(flux),
        "flux_skewness": skew(flux),
        "flux_kurtosis": kurtosis(flux),
        "flux_iqr": np.percentile(flux, 75) - np.percentile(flux, 25),
        "flux_mad": np.median(np.abs(flux - np.median(flux)))
    }
    return pd.DataFrame([features])

def detect_anomaly(df, model=None):
    features = extract_features(df)
    if model is None:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(features)
    prediction = model.predict(features)[0]
    label = "Anomaly" if prediction == -1 else "Normal"
    return label, model

def predict_future_flux(df, n_steps=10):
    from sklearn.linear_model import LinearRegression
    times = np.arange(len(df)).reshape(-1, 1)
    model = LinearRegression().fit(times, df["flux"].values)
    future_times = np.arange(len(df), len(df) + n_steps).reshape(-1, 1)
    future_flux = model.predict(future_times)
    return future_times.flatten(), future_flux

def plot_lightcurve(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["flux"], label="Flux")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Lightcurve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lightcurve_plot.png")
    plt.close()

def plot_prediction(df, future_times, future_flux):
    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["flux"], label="Flux (Original)")
    plt.plot(future_times, future_flux, label="Flux (Predicted)", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Future Flux Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicted_flux_plot.png")
    plt.close()