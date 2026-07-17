# 🛡️ ML-Powered Network Intrusion Detection System (NIDS)

A Machine Learning-based **Network Intrusion Detection System (NIDS)** developed using **Python** and the **NSL-KDD dataset**. This project uses the **Random Forest Classifier** to detect whether network traffic is **Normal** or an **Attack**. It demonstrates the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, and prediction.

---

## 📌 Features

- 📊 Dataset exploration and visualization
- 🧹 Data preprocessing and feature encoding
- 🤖 Random Forest model training
- 📈 Model evaluation using Accuracy, Precision, Recall, and F1-Score
- 📉 Confusion Matrix and Feature Importance analysis
- 🔍 Network traffic prediction (Normal / Attack)
- 💾 Model saving and loading using Joblib

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 📂 Project Structure

```
NIDS_Project/
│── 01_explore_data.py
│── 02_preprocess.py
│── 03_train_model.py
│── 04_evaluate.py
│── 05_predictor.py
│── KDDTrain+.txt
│── KDDTest+.txt
│── preprocessed_data.pkl
│── nids_model.pkl
│── requirements.txt
│── README.md
```

---

## 📊 Workflow

```
NSL-KDD Dataset
       │
       ▼
Data Exploration
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Encoding
       │
       ▼
Random Forest Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Traffic Prediction
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/ml-network-intrusion-detection-system.git
cd ml-network-intrusion-detection-system
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

---

## ▶️ Running the Project

### Step 1 – Explore Dataset

```bash
python 01_explore_data.py
```

### Step 2 – Preprocess Data

```bash
python 02_preprocess.py
```

### Step 3 – Train the Model

```bash
python 03_train_model.py
```

### Step 4 – Evaluate the Model

```bash
python 04_evaluate.py
```

### Step 5 – Predict Network Traffic

```bash
python 05_predictor.py
```

---

## 📊 Results

The Random Forest model achieved approximately:

- **Accuracy:** 99.88%
- **Precision:** 99.88%
- **Recall:** 99.88%
- **F1-Score:** 99.88%

The model successfully classifies network traffic as **Normal** or **Attack** with high accuracy using the NSL-KDD dataset.

---

## 📁 Dataset

This project uses the **NSL-KDD Dataset**, a benchmark dataset widely used for Network Intrusion Detection research.

The dataset contains:
- Network traffic records
- 41 input features
- Normal traffic
- Multiple attack categories

---

## 🎯 Future Improvements

- Real-time network traffic monitoring
- Streamlit-based web dashboard
- Multi-class attack classification
- Deep Learning models (CNN/LSTM)
- Cloud deployment

---

## 👨‍💻 Author

**Allen P Manoj**


---

