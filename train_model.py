import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
data = pd.read_csv('career_data.csv')

# Ensure that the necessary columns exist
if 'skills' not in data.columns or 'career_path' not in data.columns:
    raise ValueError("The dataset must contain 'skills' and 'career_path' columns.")

# Prepare the data
X = data['skills']
y = data['career_path']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the vectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(vectorizer, 'models/vectorizer.joblib')
joblib.dump(model, 'models/career_model.joblib')

# Evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
accuracy = model.score(X_test_vectorized, y_test)
print(f"Model accuracy: {accuracy:.2f}")
