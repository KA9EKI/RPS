import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = "dataset"
TRAIN = "train"
TEST = "test"
CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

# -------------------- Load TRAIN images --------------------
X_train, y_train = [], []
for idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_PATH, TRAIN, cls)
    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        X_train.append(img)
        y_train.append(idx)

X_train = np.array(X_train, dtype='float32') / 255.0
y_train = np.array(y_train, dtype='int32')
print("Loaded training images:", X_train.shape, y_train.shape)

# -------------------- Load TEST images --------------------
X_test, y_test = [], []
for idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_PATH, TEST, cls)
    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        X_test.append(img)
        y_test.append(idx)

X_test = np.array(X_test, dtype='float32') / 255.0
y_test = np.array(y_test, dtype='int32')

print("Loaded test images:", X_test.shape, y_test.shape)

# -------------------- Build CNN model (NO DATA AUGMENTATION) --------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Convolutional layers
    layers.Conv2D(64, (3,3), activation='relu', name='conv2d_1'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu', name='conv2d_2'),
    layers.MaxPooling2D(2,2),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------- Callbacks --------------------
chk = callbacks.ModelCheckpoint("models/best_rps_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# -------------------- Train --------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[chk, es],
    verbose=1
)

# -------------------- Evaluate --------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -------------------- Confusion Matrix --------------------
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------- Plot Curves --------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.show()

# -------------------- LIME --------------------
from lime import lime_image
from skimage.segmentation import mark_boundaries

def to_rgb(img):
    if img.shape[-1] == 1:
        return np.repeat(img, 3, axis=2)
    return img

explainer = lime_image.LimeImageExplainer()
i = 108
img_rgb = to_rgb(X_test[i]).astype('double')

explanation = explainer.explain_instance(
    img_rgb,
    model.predict,
    top_labels=3,
    hide_color=0,
    num_samples=2000
)

lime_img, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    hide_rest=False,
    num_features=10,
    min_weight=0.0
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(X_test[i])
ax[0].set_title("Original Image"); ax[0].axis("off")
ax[1].imshow(mark_boundaries(lime_img, mask))
ax[1].set_title("LIME Explanation"); ax[1].axis("off")
plt.show()

# -------------------- Save Model --------------------
model.save("models/final_rps_model.keras")

