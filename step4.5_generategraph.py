import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# --- Load dataset ---
df = pd.read_csv(r"C:\Users\abkal\Desktop\fds 4.0\landmarks_dataset.csv")  # Or landmarks_dataset_normalized.csv

X = df.drop("label", axis=1).values
y = df["label"].values

# --- Load label encoder ---
with open(r"C:\Users\abkal\Desktop\fds 4.0\label_encoderrrr_palak_wspace.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_encoded = label_encoder.transform(y)
y_categorical = to_categorical(y_encoded)

# --- Train-test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Load trained model ---
model = load_model(r"C:\Users\abkal\Desktop\fds 4.0\finger_spelling_modellll_palak_wspace.h5")

# --- Evaluate model ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"❌ Loss: {loss:.4f}")

# --- Predict ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- Confusion matrix heatmap (Orange-Yellow) ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='YlOrBr')
plt.title("Confusion Matrix (Orange-Yellow)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_orange.png")
plt.show()

# --- Classification report (as text + image) ---
report_str = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
print("📊 Classification Report:")
print(report_str)

# --- Save classification report as image ---
plt.figure(figsize=(10, 6))
plt.axis('off')
plt.title("Classification Report", fontsize=16, pad=20)
plt.text(0.01, 0.05, report_str, fontfamily='monospace', fontsize=12)
plt.tight_layout()
plt.savefig("classification_report.png", bbox_inches='tight')
plt.show()
