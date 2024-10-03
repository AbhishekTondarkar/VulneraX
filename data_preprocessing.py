# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import requests
import gzip
import json
import os
import joblib
from tqdm import tqdm
import time

def download_nvd_data(year, retries=3, timeout=60):
    url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
    
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(f"nvdcve-1.1-{year}.json.gz", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded NVD data for {year} successfully.")
            break
        except requests.exceptions.Timeout:
            print(f"Timeout occurred while downloading {year}. Retrying {attempt + 1}/{retries}...")
            if attempt + 1 == retries:
                print(f"Failed to download NVD data for {year} after {retries} attempts.")
                raise
            time.sleep(2)  # Wait before retrying
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error occurred: {e}. Retrying {attempt + 1}/{retries}...")
            if attempt + 1 == retries:
                print(f"Failed to download NVD data for {year} after {retries} attempts.")
                raise
            time.sleep(2)
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"Chunked encoding error while downloading {year}: {e}. Retrying {attempt + 1}/{retries}...")
            if attempt + 1 == retries:
                print(f"Failed to download NVD data for {year} after {retries} attempts.")
                raise
            time.sleep(2)

def extract_nvd_data(year):
    with gzip.open(f"nvdcve-1.1-{year}.json.gz", "rb") as f:
        data = json.load(f)
    return data["CVE_Items"]

def preprocess_nvd_data(items):
    processed_data = []
    for item in items:
        cve_id = item["cve"]["CVE_data_meta"]["ID"]
        description = item["cve"]["description"]["description_data"][0]["value"]
        
        if "baseMetricV3" in item["impact"]:
            severity = item["impact"]["baseMetricV3"]["cvssV3"]["baseSeverity"]
            score = item["impact"]["baseMetricV3"]["cvssV3"]["baseScore"]
        elif "baseMetricV2" in item["impact"]:
            severity = item["impact"]["baseMetricV2"]["severity"]
            score = item["impact"]["baseMetricV2"]["cvssV2"]["baseScore"]
        else:
            severity = "UNKNOWN"
            score = 0
        
        cwe = []
        if "problemtype" in item["cve"]:
            for pt in item["cve"]["problemtype"]["problemtype_data"]:
                for desc in pt["description"]:
                    if desc["value"].startswith("CWE-"):
                        cwe.append(desc["value"])
        
        processed_data.append({
            "cve_id": cve_id,
            "description": description,
            "severity": severity,
            "score": score,
            "cwe": cwe
        })
    
    return pd.DataFrame(processed_data)

def feature_engineering(df, max_features=1000):
    # Text features
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    text_features = tfidf.fit_transform(df['description'])
    
    # CWE features
    mlb = MultiLabelBinarizer()
    cwe_features = mlb.fit_transform(df['cwe'])
    
    # Combine features
    features = np.hstack((text_features.toarray(), cwe_features))
    
    # Handle missing or unknown severity
    df['severity'] = df['severity'].replace({'UNKNOWN': 'LOW'})  # Replace UNKNOWN with 'LOW'
    df['severity'] = df['severity'].map({'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}).fillna(0)  # Ensure no NaN values
    
    labels = df['severity']
    
    return features, labels, tfidf, mlb

def prepare_data(start_year=2020, end_year=2023, sample_size=10000, delete_existing_data=False):
    all_data = []
    for year in tqdm(range(start_year, end_year + 1)):
        if os.path.exists(f"nvdcve-1.1-{year}.json.gz"):
            if delete_existing_data:
                os.remove(f"nvdcve-1.1-{year}.json.gz")
                print(f"Deleted existing NVD data for {year}.")
        else:
            download_nvd_data(year)
        
        items = extract_nvd_data(year)
        all_data.extend(items)
    
    df = preprocess_nvd_data(all_data)
    
    # Sample the data
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    features, labels, tfidf, mlb = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tfidf, mlb

if __name__ == "__main__":
    # Set delete_existing_data to True if you want to delete previously downloaded NVD data files
    delete_existing_data = False
    X_train, X_test, y_train, y_test, tfidf, mlb = prepare_data(delete_existing_data=delete_existing_data)
    
    # Ensure the directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # Save the preprocessed data
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    joblib.dump(tfidf, "model/tfidf_vectorizer.joblib")
    joblib.dump(mlb, "model/cwe_binarizer.joblib")
    
    print("Data preprocessing completed.")
