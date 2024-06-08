import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import pickle


def load_label_encoders():
    # Load LabelEncoders
    with open('./datasets/label_encoders.pkl', 'rb') as f:
        loaded_label_encoders = pickle.load(f)

    return loaded_label_encoders


def load_data():
    # load prepared dataset
    X = pd.read_csv('./datasets/X.csv')
    y = pd.read_csv('./datasets/y.csv')
    weights = pd.read_csv('./datasets/weights.csv')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def build_tf_model():
    # Build a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, weights_train, weights_test = load_data()
    label_encoders = load_label_encoders()


    model = build_tf_model()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  weighted_metrics=['accuracy'])

    # Train the model with sample weights
    model.fit(X_train, y_train, sample_weight=weights_train, epochs=30, batch_size=32, validation_split=0.2)

    # Predict probabilities on the test set
    pred_probs = model.predict(X_test)

    # Apply correction factors to predictions
    corrected_predictions = np.zeros_like(pred_probs)
    for i in range(pred_probs.shape[0]):
        correction_factor = 1 / weights_test.iloc[i]  # Correcting the correction factor
        for j in range(pred_probs.shape[1]):
            p = pred_probs[i, j]
            corrected_predictions[i, j] = (p * correction_factor) / (p * correction_factor + (1 - p))

    # Normalize the corrected predictions
    corrected_predictions = corrected_predictions / corrected_predictions.sum(axis=1, keepdims=True)

    # Final predictions
    final_predictions = np.argmax(corrected_predictions, axis=1)

    # Display corrected classification report
    print(classification_report(y_test, final_predictions, target_names=label_encoders['parents_education'].classes_))



