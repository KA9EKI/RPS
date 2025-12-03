import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow import keras
import cv2
import os

# ------------------- CONFIG -------------------
MODEL_PATH = "models/improved_rps_model.keras"
DATA_PATH = "dataset/test"
CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = (64, 64)

# ------------------- LOAD MODEL -------------------
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------- LOAD A SAMPLE IMAGE -------------------
def load_image(class_name):
    """Loads one random image from the given class folder."""
    folder = os.path.join(DATA_PATH, class_name)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    img_path = os.path.join(folder, np.random.choice(files))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    return img, img_resized / 255.0

# Pick one class to explain
chosen_class = np.random.choice(CLASSES)
orig_img, input_img = load_image(chosen_class)
print(f"🖼️ Explaining prediction for class: {chosen_class}")

# ------------------- LIME EXPLAINER -------------------
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    input_img.astype('double'),
    classifier_fn=lambda x: model.predict(x),
    top_labels=3,
    hide_color=0,
    num_samples=1000
)

# ------------------- SHOW RESULTS -------------------
top_label = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(
    top_label,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(input_img)
plt.title(f"Original ({chosen_class})")

plt.subplot(1,2,2)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f"LIME Explanation ({CLASSES[top_label]})")

plt.tight_layout()
plt.show()
