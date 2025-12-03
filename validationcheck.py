import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# 1. Load Model & Classes
# ----------------------------
MODEL_PATH = "models/final_rps_model.keras"
CLASSES = ["rock", "paper", "scissors"]

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

IMG_SIZE = (64, 64)
img = cv2.imread("scissors5.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)  # shape: (1,64,64,3)

preds = model.predict(img, verbose=0)
pred_class = np.argmax(preds)
pred_label = CLASSES[pred_class]
confidence = preds[0][pred_class] * 100

print(preds,pred_class,confidence,pred_label)