# 🌌 StellarScope

**StellarScope** is a compact Python-based tool for analyzing Kepler lightcurve data. It performs anomaly detection and future flux prediction using machine learning, and provides an interactive Gradio web interface for easy use and visualization.

## 🚀 Features

- 📈 Extracts statistical features from NASA Kepler lightcurve `.fits` or `.csv` files
- 🧠 Detects anomalies in lightcurve flux using Isolation Forest
- 🔮 Predicts future flux values with a trained Random Forest model
- 🖼️ Visualizes uploaded lightcurves interactively
- 🖱️ Includes a user-friendly Gradio web app

## 🛠 Installation

pip install -r requirements.txt
📂 Project Structure
bash
Copy
Edit
├── app.py                # Gradio interface
├── model.py              # Anomaly detection & ML model training
├── utils.py              # Feature extraction and plotting utilities
├── requirements.txt      # Required packages
├── example.csv           # Sample lightcurve input
└── README.md             # This file
💻 Usage
To run the Gradio app:
Copy
Edit
python app.py
Then open the local URL provided in your browser and upload a .csv file with lightcurve data (columns: time, flux, flux_err).

🧪 Example Input Format
csv
Copy
Edit
time,flux,flux_err
260.2154571,0.99961776,0.000405104
260.2161383,0.99889493,0.000405024
...
📊 Outputs
Anomaly prediction (Normal / Anomalous)

Visualized lightcurve plot

Forecasted future flux values

📎 License
MIT License

🙌 Acknowledgements
NASA Exoplanet Archive (Kepler Data)

Lightkurve

Scikit-learn

Gradio
