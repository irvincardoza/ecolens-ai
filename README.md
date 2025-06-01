# ♻️ EcoLens AI – Real-Time Trash Detection for Smarter Cities

EcoLens AI is a lightweight, real-time waste classification tool that helps users identify common trash types (plastic, paper, metal, etc.) using their webcam. It's designed for schools, communities, and smart homes aiming to improve recycling habits through AI.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-red)
![License](https://img.shields.io/badge/license-MIT-green)


---

## 🧠 What It Does

- 🧪 Classifies objects as `plastic`, `metal`, `glass`, `paper`, `cardboard`, or `trash`
- 📦 Uses a fixed **center-box detection zone** to reduce false positives
- 🤖 Combines two models (CNN + MobileNetV2) for better accuracy
- 🖥️ Runs in real-time using your webcam (OpenCV)
- ⚡ Trained on public Kaggle dataset of labeled waste images

---

## 🚀 How to Run

### 💻 Local (Python)
1. Clone this repo  
   ```bash
   git clone https://github.com/yourusername/ecolens-ai.git

   ```

2. Install dependencies  
   ```bash
   pip install tensorflow opencv-python
   ```

3. Run webcam detection  
   ```bash
   python app.py
   ```

> Make sure `waste_cnn_classifier.h5` and `waste_classifier_mobilenetv2.h5` are in the same directory.

---

## 🧱 Project Structure

```
ecolens-ai/
├── Main.ipynb                 ← Kaggle training notebook
├── app.py                     ← Real-time webcam detection
├── waste_cnn_classifier.h5    ← Trained CNN model
├── waste_classifier_mobilenetv2.h5 ← MobileNetV2 model
├── README.md
```

---

## 📸 Example Output

![output-demo](https://your-screenshot-link.com)  
> Center-box detects object → predicts "Plastic (88%)" → shows bounding box

---

## 🛠️ Built With

- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- Custom CNN
- OpenCV
- Jupyter + Kaggle

---

## ✅ Accuracy

| Model          | Validation Accuracy |
|----------------|----------------------|
| CNN            | 62.4%                |
| MobileNetV2    | 71.3%                |
| **Ensemble**   | **~73.5%**           |

---

## 🧪 Future Features

- [ ] YOLOv7 integration for full-frame detection
- [ ] Voice assistant integration (“This goes in recycling”)
- [ ] Raspberry Pi + webcam smart bin prototype
- [ ] Gradio or Streamlit web UI

---

## 📜 License

MIT License. Free to use, modify, and distribute.

---

## 🤝 Contributions Welcome

Feel free to open issues or submit pull requests if you’d like to improve detection, retrain on new waste types, or make the system more interactive.

---

## 👥 Team

Built during Hack the Bronx 2025  
[Your Name] • [Your GitHub] • [Optional: LinkedIn]
