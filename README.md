# 🩺 Pneumonia Detection Using Deep Learning

An AI-powered web application that detects **Pneumonia** from Chest X-ray images using a **DenseNet121** deep learning model. The application is built with **Flask** and provides predictions through a simple web interface.

---

## 📌 Features

- Upload Chest X-ray images
- Detect whether the patient has **Pneumonia** or **Normal**
- Deep Learning model based on DenseNet121
- User-friendly Flask web interface
- Grad-CAM visualization for model explainability
- Responsive frontend

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- DenseNet121
- Flask
- OpenCV
- NumPy
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```
DL/
│── app.py
│── gradcam.py
│── check_model.py
│── requirements.txt
│── pneumonia_densenet121.keras
│
├── templates/
├── static/
├── chest_xray/
│   ├── train/
│   ├── test/
│   └── val/
│
└── saved_model_v3/
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/vignapan7266/Pneumonia-Detection.git
```

### Move into the project

```bash
cd Pneumonia-Detection
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 🧠 Model

- Architecture: DenseNet121
- Framework: TensorFlow/Keras
- Input Size: 224 × 224
- Classes:
  - Normal
  - Pneumonia

---

## 📊 Dataset

The project uses the **Chest X-Ray Pneumonia Dataset** from Kaggle.

Due to GitHub file size limitations, the dataset is **not included** in this repository.

Dataset Link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 📸 Screenshots

### Home Page

_Add screenshot here_

### Prediction Result

_Add screenshot here_

### Grad-CAM Visualization

_Add screenshot here_

---

## 📈 Future Improvements

- Improve model accuracy
- Deploy on Render or Railway
- User authentication
- Database integration
- Medical report generation
- Multi-class lung disease detection

---

## 👨‍💻 Author

**Vignapan**

GitHub: https://github.com/vignapan7266

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.