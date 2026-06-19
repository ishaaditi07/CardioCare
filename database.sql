CREATE DATABASE heart_disease_db;

USE heart_disease_db;

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    sex VARCHAR(20),
    cp VARCHAR(50),
    trestbps INT,
    chol INT,
    fbs VARCHAR(10),
    restecg VARCHAR(50),
    thalach INT,
    exang VARCHAR(10),
    oldpeak DECIMAL(4,1),
    slope VARCHAR(30),
    ca VARCHAR(10),
    thal VARCHAR(50),
    predicted_severity INT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);