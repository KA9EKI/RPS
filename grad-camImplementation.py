import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models

# ------------------ Constants ------------------
MODEL_PATH = "models/improved_rps_model.keras"
IMG_PATH = "./dataset/train/scissors/scissors01-003.png"
IMG_SIZE = (64, 64)
CLASSES = ["rock", "paper", "scissors"]

# ------------------ Rebuild Functional Model ------------------
def build_functional_rps(input_shape=(64,64,3), n_classes=3):
    inputs = keras.Input(shape=input_shape, name="input_layer")

    x = layers.Conv2D(64, (3,3), activation='relu', name='conv2d_1')(inputs)
    x = layers.MaxPooling2D(2,2)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', name='conv2d_2')(x)
    x = layers.MaxPooling2D(2,2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RPS_Functional")
    return model

# ------------------ Load weights ------------------
saved_model = keras.models.load_model(MODEL_PATH)
print("✅ Original Sequential model loaded")

functional_model = build_functional_rps((64,64,3), 3)
functional_model.set_weights(saved_model.get_weights())
print("🔁 Functional model built and weights copied")

# ------------------ Preprocess Image ------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"❌ Image not found at: {IMG_PATH}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img_input = np.expand_dims(img.astype("float32") / 255.0, axis=0)

# ------------------ Grad-CAM ------------------
def make_gradcam_heatmap(model, img_array, last_conv_name):
    last_conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("❌ Gradients are None — check layer connectivity.")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap, int(pred_index)  # ✅ removed .numpy()

# ------------------ Run Grad-CAM ------------------
last_conv_layer_name = "conv2d_2"
print("✅ Last Conv Layer:", last_conv_layer_name)

heatmap, pred_idx = make_gradcam_heatmap(functional_model, img_input, last_conv_layer_name)
pred_label = CLASSES[pred_idx]
print(f"✅ Predicted class: {pred_label}")

# ------------------ Overlay Heatmap ------------------
heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)

cv2.imwrite("gradcam_overlay.jpg", overlay)
print("✅ Grad-CAM saved as gradcam_overlay.jpg")

# ------------------ Display ------------------
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(heatmap, cmap="jet"); plt.title("Grad-CAM Heatmap"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Overlay ({pred_label})"); plt.axis("off")
plt.tight_layout(); plt.show()
