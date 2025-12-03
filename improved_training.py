import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = "dataset"
TRAIN = "train"
TEST = "test"
CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

# ------------------- Load Dataset -------------------
def load_images(folder_path, classes):
    X, y = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(folder_path, cls)
        files = os.listdir(folder)
        for fname in files:
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(folder, fname))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                X.append(img)
                y.append(idx)
    return np.array(X)/255.0, np.array(y)

X_train, y_train = load_images(os.path.join(DATA_PATH, TRAIN), CLASSES)
X_test, y_test = load_images(os.path.join(DATA_PATH, TEST), CLASSES)

print("Train:", X_train.shape)
print("Test:", X_test.shape)


# ------------------- VERY STABLE MODEL -------------------
model = models.Sequential([
    layers.Input((64, 64, 3)),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0004),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------- Callbacks -------------------
chk = callbacks.ModelCheckpoint(
    "models/improved_model.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# ------------------- Train -------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[chk, es],
    verbose=1
)

# ------------------- Evaluate -------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {acc*100:.2f}%")

# 6) Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict on test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report
print("\n\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 7) Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.show()

model.save("models/improved_rps_model.keras")
