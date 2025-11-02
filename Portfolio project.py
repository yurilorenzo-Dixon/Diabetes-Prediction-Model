import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_pima():
    
    csv_paths = ['pima-indians-diabetes.csv', 'diabetes.csv', 'pima.csv']
    for p in csv_paths:
        if os.path.exists(p):
            print(f" Loading local CSV file: {p}")
            return pd.read_csv(p)

    
    try:
        print(" Attempting to fetch dataset from OpenML (ID 43483)...")
        df = fetch_openml(data_id=43483, as_frame=True)
        if hasattr(df, 'frame') and df.frame is not None:
            print("Successfully loaded Pima dataset from OpenML (ID 43483)")
            return df.frame
        else:
            print(" Fetch succeeded but returned data differently; building DataFrame manually.")
            data = pd.DataFrame(df.data, columns=df.feature_names)
            data['class'] = df.target
            return data
    except Exception as e:
        print("Could not load dataset automatically. Please download 'pima-indians-diabetes.csv' and place it next to this script.")
        raise e

df = load_pima()
print("Dataset shape:", df.shape)
print(df.head())


if 'Outcome' in df.columns:
    label_col = 'Outcome'
elif 'Class' in df.columns:
    label_col = 'Class'
elif 'class' in df.columns:
    label_col = 'class'
elif 'diabetes' in df.columns:
    label_col = 'diabetes'
else:
    label_col = df.columns[-1]


if df[label_col].dtype == object:
    df[label_col] = df[label_col].map({'pos': 1, 'neg': 0}).astype(int)


maybe_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
present_zero_cols = [c for c in maybe_zero_cols if c in df.columns]
for c in present_zero_cols:
    df[c] = df[c].replace(0, np.nan)
    median = df[c].median()
    df[c] = df[c].fillna(median)


X = df.drop(columns=[label_col])
y = df[label_col].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


input_dim = X_train_scaled.shape[1]

def make_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

model = make_model(input_dim)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)


model.save('pima_model_final.h5')


test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
y_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f" Test accuracy: {test_acc:.4f}, Test AUC: {auc:.4f}")


print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()


RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title('ROC Curve')
plt.savefig('roc_curve.png')
plt.close()


plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History - Loss')
plt.savefig('history_loss.png')
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History - Accuracy')
plt.savefig('history_acc.png')
plt.close()

pred_df = pd.DataFrame({
    'true': y_test.values,
    'pred_proba': y_prob,
    'pred_label': y_pred
})
pred_df.to_csv('test_predictions.csv', index=False)

print("\n🎉 All artifacts saved:")
print(" - pima_model_final.h5")
print(" - best_model.h5")
print(" - confusion_matrix.png")
print(" - roc_curve.png")
print(" - history_loss.png")
print(" - history_acc.png")
print(" - test_predictions.csv\n")
