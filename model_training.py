import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_neural_network(X_train, y_train):
    # Define the Keras neural network model
    nn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # Adjust this based on your number of classes
    ])
    
    nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    return nn

def evaluate_model(model, X_test, y_test, is_rf=False):
    if is_rf:
        # For Random Forest, directly use predictions
        y_pred_classes = model.predict(X_test)  # Returns a 1D array of class labels
    else:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    
    print(classification_report(y_test, y_pred_classes))
    print(confusion_matrix(y_test, y_pred_classes))

if __name__ == "__main__":
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Training Neural Network...")
    nn_model = train_neural_network(X_train, y_train)
    
    print("Random Forest Evaluation:")
    evaluate_model(rf_model, X_test, y_test, is_rf=True)  # Pass is_rf=True for Random Forest
    
    print("Neural Network Evaluation:")
    evaluate_model(nn_model, X_test, y_test)  # For Neural Network, is_rf=False by default
    
    joblib.dump(rf_model, "model/random_forest_model.joblib")
    nn_model.save("model/neural_network_model.h5")  # Save the neural network model in .h5 format
    
    print("Model training completed.")
