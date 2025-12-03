import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# 1. Load Model & Classes
# ----------------------------
MODEL_PATH = "models/improved_rps_model.keras"
CLASSES = ["rock", "paper", "scissors"]

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

IMG_SIZE = (64, 64)

# ----------------------------
# 2. Initialize Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ----------------------------
# 3. Detection Loop
# ----------------------------
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Flip the frame horizontally for mirror-like view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Define a square region of interest (ROI)
    roi_size = 300
    x1, y1 = w//2 - roi_size//2, h//2 - roi_size//2
    x2, y2 = w//2 + roi_size//2, h//2 + roi_size//2
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for prediction
    img = cv2.resize(roi, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1,64,64,3)

    # Predict
    preds = model.predict(img, verbose=0)
    pred_class = np.argmax(preds)
    pred_label = CLASSES[pred_class]
    confidence = preds[0][pred_class] * 100

    # Display results
    cv2.putText(frame, f"Prediction: {pred_label} ({confidence:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Place your hand here", (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Rock Paper Scissors Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 4. Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
