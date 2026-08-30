import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Recall, AUC

def load_data(input_file='research/prepared_data.pkl'):
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def train_xgboost(data):
    print("\n" + "="*50)
    print("TRAINING BASELINE XGBOOST (2D Flattened)")
    print("="*50)
    
    X_train, y_train = data['X_train_flat'], data['y_train']
    X_test, y_test   = data['X_test_flat'], data['y_test']
    scale_weight     = data['scale_pos_weight']

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    
    # Predict probabilities for PR-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"XGBoost Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    
    return model

def train_lstm(data):
    print("\n" + "="*50)
    print("TRAINING BASELINE LSTM (3D Sequential)")
    print("="*50)
    
    X_train, y_train = data['X_train_3d'], data['y_train']
    X_test, y_test   = data['X_test_3d'], data['y_test']
    
    n_timesteps = X_train.shape[1]  # Should be 10
    n_features = X_train.shape[2]

    # Calculate exact class weights for Keras
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    total = len(y_train)
    
    weight_0 = (1 / n_neg) * (total / 2.0)
    weight_1 = (1 / n_pos) * (total / 2.0)
    class_weight = {0: weight_0, 1: weight_1}

    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(n_timesteps, n_features)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[Recall(name='recall'), AUC(curve='PR', name='pr_auc')]
    )

    print("Fitting LSTM Model...")
    # Suppress verbose output for clean logs
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=256,
        class_weight=class_weight,
        validation_split=0.1,
        verbose=1
    )

    print("\nEvaluating LSTM on Test Set...")
    results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"LSTM Results:")
    print(f"  Loss:      {results[0]:.4f}")
    print(f"  Recall:    {results[1]:.4f}")
    print(f"  PR-AUC:    {results[2]:.4f}")

    return model

if __name__ == "__main__":
    # Ensure TF doesn't clutter the console with hardware warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    if not os.path.exists('research/prepared_data.pkl'):
        print("Error: Prepared data not found. Run data_preparation.py first.")
    else:
        data = load_data()
        train_xgboost(data)
        train_lstm(data)
