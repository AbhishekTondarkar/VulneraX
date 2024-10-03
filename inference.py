import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

rf_model = joblib.load("model/random_forest_model.joblib")
nn_model = tf.keras.models.load_model("model/neural_network_model.h5")
tfidf = joblib.load("model/tfidf_vectorizer.joblib")
mlb = joblib.load("model/cwe_binarizer.joblib")

def preprocess_input(code):
    text_features = tfidf.transform([code]).toarray()
    cwe_features = mlb.transform([[]])  # Empty CWE for new code
    features = np.hstack((text_features, cwe_features))
    return features

def get_vulnerability_type(cwe_probabilities):
    top_cwes = mlb.classes_[np.argsort(cwe_probabilities)[-3:]]
    return ", ".join(top_cwes)

def get_severity(severity_index):
    severity_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    return severity_map[severity_index]

def analyze_vulnerability(code):
    features = preprocess_input(code)
    
    rf_prediction = rf_model.predict(features)[0]
    rf_probabilities = rf_model.predict_proba(features)[0]
    
    nn_prediction = np.argmax(nn_model.predict(features), axis=1)[0]
    nn_probabilities = nn_model.predict(features)[0]
    
    # Ensemble the predictions
    ensemble_probabilities = (rf_probabilities + nn_probabilities) / 2
    ensemble_prediction = np.argmax(ensemble_probabilities)
    
    vulnerability_type = get_vulnerability_type(ensemble_probabilities)
    severity = get_severity(ensemble_prediction)
    confidence = ensemble_probabilities[ensemble_prediction]
    
    return {
        "is_vulnerable": ensemble_prediction > 0,
        "vulnerability_type": vulnerability_type,
        "severity": severity,
        "confidence": confidence,
        "repair_suggestion": generate_repair_suggestion(vulnerability_type, severity)
    }

def generate_repair_suggestion(vulnerability_type, severity):
    # This function would contain a database of repair suggestions based on vulnerability type and severity
    # For simplicity, we'll return a generic suggestion
    return f"Review and fix potential {vulnerability_type} vulnerabilities. Given the {severity} severity, prioritize this issue."