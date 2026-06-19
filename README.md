

# 🫀 CardioCare : Heart Disease Severity Prediction System

## 📌 Project Overview

This project is a web-based system that predicts the severity of heart disease based on patient medical input. It uses a trained machine learning model with a Flask backend and a simple HTML frontend. Predictions are also stored in a MySQL database.

---

## 📂 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                         # Flask backend API
├── heart_disease_prediction.py    # Model training script
├── Heart_disease_cleveland_new.csv# Dataset
├── database.sql                   # MySQL table creation script
├── index.html                    # Frontend UI
```

---

## ⚙️ Technologies Used

* Python
* Flask
* Pandas
* Scikit-learn
* MySQL
* HTML, CSS, JavaScript

---

## 🧠 Model Workflow

1. Load dataset (`Heart_disease_cleveland_new.csv`)
2. Clean and preprocess data
3. Encode categorical values
4. Handle missing values
5. Scale features using StandardScaler
6. Balance dataset using SMOTE
7. Train Logistic Regression model
8. Send trained model output to Flask API

---

## 🖥️ Features

* User-friendly web form (`index.html`)
* Takes patient health inputs
* Sends data to Flask API (`/predict`)
* Predicts heart disease severity (0–4)
* Stores prediction in MySQL database (`database.sql`)

---

## 🗄️ Database Setup

Run this SQL file in MySQL:

```sql id="c8k3qv"
database.sql
```

It will create a table to store:

* Patient details
* Prediction result
* Timestamp

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash id="k2q9lm"
pip install flask flask-cors pandas scikit-learn imbalanced-learn mysql-connector-python joblib
```

---

### 2. Setup MySQL database

* Create database: `heart_disease_db`
* Run `database.sql`

---

### 3. Run backend

```bash id="r8w2dp"
python app.py
```

---

### 4. Open frontend

Open:

```
index.html
```

---

## 📥 API Endpoint

### POST `/predict`

#### Input:

Patient health parameters (age, sex, chest pain type, etc.)

#### Output:

```json id="v4n7zx"
{
  "predicted_severity": 2
}
```

---

## 📊 Severity Levels

* 0 → Low risk
* 1 → Mild risk
* 2 → Moderate risk
* 3 → High risk
* 4 → Very high risk

---

## ⚠️ Notes

* Input values must match training format exactly
* Categorical values must match dataset labels
* Flask must be running before submitting form
