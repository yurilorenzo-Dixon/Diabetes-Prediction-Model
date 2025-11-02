
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import save_model
from sklearn.datasets import fetch_openml


def load_pima():
    try:
        print(" Attempting to load Pima dataset from OpenML (ID 43483)...")
        df = fetch_openml(data_id=43483, as_frame=True).frame
        print(" Successfully loaded Pima dataset from OpenML (ID 43483).")
        return df
    except Exception as e:
        print(" Automatic download failed. Please download manually from:")
        print("   https://www.openml.org/d/43483")
        raise e


df = load_pima()


possible_targets = ['class', 'diabetes', 'target', 'Diabetes_binary', 'Outcome']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    raise ValueError(f" Could not find target column. Available columns: {list(df.columns)}")

print(f" Target column detected: '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col]

y = y.apply(lambda x: 1 if str(x).lower() in ['tested_positive', 'pos', 'yes', '1', 'true'] else 0)

print(" Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Data split complete.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(" Data scaling complete.")


print(" Building model...")

model = Sequential([
    Input(shape=(X_train.shape[1],)),  
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(" Model compiled successfully.")


print(" Training the model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
print(" Training complete.")


print(" Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy: {accuracy:.2f}")


predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
print(" Predictions complete.")
print(f" Sample predictions: {predictions[:10]}")


model.save('diabetes_model.keras')  
print(" Model saved successfully as 'diabetes_model.keras'.")


print(" Portfolio Project completed successfully!")

