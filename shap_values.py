# import shap
# import pandas as pd
# import joblib

# # Load model and data
# model = joblib.load("seizft_xgb_model.pkl")
# df = pd.read_csv("datasets/eeg_features_baseline.csv")

# # Prepare features
# X = df.drop(columns=["seizure", "recording", "channel", "epoch", "second", "augmentation"])

# # Create SHAP explainer and compute values
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# # Summary plot
# shap.summary_plot(shap_values, X)

#####################################################################################

# import os
# import shap
# import pandas as pd
# import joblib

# # Filenames for saved data
# SHAP_FILE = "shap_values_seizure.pkl"
# X_FILE = "X_scaled_df.pkl"

# # Load model and scaler
# model = joblib.load("rf_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Load and prepare the data
# df = pd.read_csv("datasets/eeg_features_toy.csv")
# X = df.drop(columns=["seizure", "recording", "epoch", "channel", "second", "alpha_peak"])
# X_scaled = scaler.transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # Check if SHAP values and data already exist
# if os.path.exists(SHAP_FILE) and os.path.exists(X_FILE):
#     print("Loading precomputed SHAP values...")
#     shap_vals = joblib.load(SHAP_FILE)
#     X_scaled_df = joblib.load(X_FILE)
# else:
#     print("Computing SHAP values, this might take a moment...")

#     # SHAP explainability
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_scaled_df)

#     # Handle binary classification shape
#     shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

#     # Save results
#     joblib.dump(shap_vals, SHAP_FILE)
#     joblib.dump(X_scaled_df, X_FILE)
#     print("SHAP values saved.")

# # Plot
# shap.summary_plot(shap_vals, X_scaled_df, class_names=["No Seizure", "Seizure"])

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Optional: enable JS visualization if using Jupyter
# shap.initjs()

# File paths
SHAP_FILE = "shap_values_seizure.pkl"
X_FILE = "X_scaled_df.pkl"

# Load saved SHAP values and data
shap_vals = joblib.load(SHAP_FILE)           # shape: (n_samples, n_features, 2)
X_scaled_df = joblib.load(X_FILE)            # shape: (n_samples, n_features)

# Print shape to verify
print("SHAP shape:", shap_vals.shape)
print("Data shape:", X_scaled_df.shape)

# --- Extract SHAP values for class 1 (e.g., "Seizure") ---
# shap_vals has shape (n_samples, n_features, 2)
shap_vals_class1 = shap_vals[:, :, 1]

# --- Summary Plot (dot plot) ---
shap.summary_plot(shap_vals_class1, X_scaled_df)

# --- Summary Bar Plot ---
shap.summary_plot(shap_vals_class1, X_scaled_df, plot_type="bar")

# --- Force Plot (for one instance) ---
# Optionally estimate expected value (e.g., mean predicted probability for class 1)
# If not available, set to 0 or calculate from model.predict_proba(X)
expected_value = shap_vals_class1.mean()

# Choose an index to explain
i = 0
plt.figure()
shap.force_plot(
    base_value=expected_value,
    shap_values=shap_vals_class1[i],
    features=X_scaled_df.iloc[i],
    feature_names=X_scaled_df.columns,
    matplotlib=True  # use matplotlib backend
)
plt.show()

# --- Waterfall Plot (modern, cleaner version for one sample) ---
from shap import Explanation

explanation = Explanation(
    values=shap_vals_class1[i],
    base_values=expected_value,
    data=X_scaled_df.iloc[i],
    feature_names=X_scaled_df.columns
)

shap.plots.waterfall(explanation)



