import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # for saving the model

# Load dataset
df = pd.read_csv("datasets/eeg_features_baseline.csv")

# Features and labels
X = df.drop(columns=["seizure", "recording", "channel", "epoch", "second", "augmentation"])
y = df["seizure"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

# Save the model
joblib.dump(model, "seizft_xgb_model.pkl")
print("Model saved to seizft_xgb_model.pkl")
