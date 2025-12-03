# ✋ Real-Time Hand Gesture Recognition using CNN and Explainable AI

This project implements a **real-time Rock–Paper–Scissors hand gesture recognition system** using a **Convolutional Neural Network (CNN)** integrated with **Explainable AI (XAI)** techniques — **Grad-CAM** and **LIME** — to visualize and interpret model predictions.
The system takes live webcam input and predicts gestures while highlighting regions responsible for classification, increasing transparency and user trust.

---

## 🚀 Features

* Rock / Paper / Scissors gesture classification
* Live webcam inference using **OpenCV**
* **Grad-CAM heatmaps** for class-specific activated regions
* **LIME explanations** for pixel-level feature importance
* Training + Validation graphs and confusion matrix visualization
* High prediction accuracy with real-time performance

---

## 📂 Project Structure

```
├── dataset/                ← Kaggle dataset (Rock–Paper–Scissors)
├── improved_training.py    ← CNN training script
├── grad-camImplementation.py ← Grad-CAM explanation generation
├── limeImplimentation.py   ← LIME explanation generation
├── detect_rps.py           ← Real-time webcam prediction system
├── models/                 ← Trained model (.h5)
├── outputs/                ← Saved heatmaps, LIME results, graphs
├── images/                 ← IEEE paper figures
└── README.md
```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies:

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
lime
```

---

## 📊 Model Performance

| Metric              | Score      |
| ------------------- | ---------- |
| Training Accuracy   | **99.95%** |
| Validation Accuracy | **88.90%** |

Visual performance insights:

* Confusion matrix
* Loss–accuracy curves
* Grad-CAM & LIME visualizations

---

## 🧠 Explainable AI

| Technique    | Purpose                                                      |
| ------------ | ------------------------------------------------------------ |
| **Grad-CAM** | Visualizes spatial regions influencing the CNN prediction    |
| **LIME**     | Shows pixel-segment contributions affecting model confidence |

Both methods validate that the model focuses on gesture shape rather than background.

---

## 🎥 Real-Time Deployment

Run the live gesture detection system:

```bash
python detect_rps.py
```

This script:

1. Captures webcam feed
2. Extracts and preprocesses a hand region of interest (ROI)
3. Predicts gesture using trained CNN
4. Overlays label + (optional) Grad-CAM heatmap on screen

---

## 📥 Dataset Source

The dataset is publicly available on Kaggle:
🔗 [https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)

---

## 👥 Team Members

| Name                | Roll No.   | Contribution                                        |
| ------------------- | ---------- | --------------------------------------------------- |
| **Mohd Arham**      | 1024240051 | CNN training, Grad-CAM, real-time deployment        |
| **Sidak Raj Virdi** | 1024240043 | Hyperparameter tuning, LIME, graphs & documentation |

Both team members contributed equally to research, debugging, testing, and report writing.

---

## 🔮 Future Improvements

* Expand gesture set and include **dynamic gestures**
* Deploy model on **Raspberry Pi / Edge devices**
* Integrate with **robotic or gaming control interfaces**
* Improve robustness to background clutter and motion blur

---

## 📜 License

This project is for **academic and research purposes**.
Feel free to fork and extend.

---

