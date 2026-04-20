# 🧠 NeuroScan AI

**NeuroScan AI** is a deep learning-powered web application that detects brain tumors from MRI scans using a trained Xception-based model. The app provides predictions, confidence scores, visual explanations (Grad-CAM), and downloadable diagnostic reports.

---

## 🚀 Features

* 🧠 Brain tumor classification (4 classes)
* 📊 Confidence score for predictions
* 🔥 Grad-CAM heatmap visualization (AI attention)
* 📄 Downloadable PDF diagnostic report
* 🌐 Interactive web interface using Streamlit

---

## 🧠 Model Details

* **Architecture:** Xception (via timm)
* **Framework:** PyTorch
* **Input Size:** 299 × 299
* **Classes:**

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor

---

## 📂 Project Structure

```
NeuroScan-AI/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
```

> ⚠️ Note: The trained model file (`.pth`) is not included due to GitHub size limits.

---

## 📥 Model Weights

The model is automatically downloaded when you run the application.

If needed, you can manually download it from:
👉 https://drive.google.com/file/d/19mGV7eKw8oMSQxwtypWQtbyfHv_iNJFQ

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroScan-AI.git
cd NeuroScan-AI
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📸 Demo

*(Add screenshots of your UI here for better presentation)*

---

## 🔍 How It Works

1. Upload an MRI scan image
2. The model processes the image
3. Predicts tumor type
4. Displays:

   * Prediction result
   * Confidence score
   * AI attention heatmap
5. Generate and download a report

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
It is **not a medical diagnostic tool** and should not be used for real-world medical decisions.

---

## 🛠️ Tech Stack

* Streamlit
* PyTorch
* timm
* OpenCV
* NumPy
* Matplotlib
* ReportLab

---

## 🌟 Future Improvements

* Deploy live (Streamlit Cloud / Hugging Face Spaces)
* Improve model accuracy with larger datasets
* Add multi-scan comparison
* Mobile-friendly UI

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out.

---

⭐ If you like this project, consider giving it a star!
