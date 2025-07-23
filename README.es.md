# SALARY PREDICTOR

# 🧠 IT Salary Prediction from Job Descriptions (U.S. Tech Sector)

A complete end-to-end **machine learning project** for predicting salaries in the U.S. IT industry using **natural language processing** on real job description texts.

This project combines **data extraction**, **web scraping**, **NLP (BERT)**, **feature engineering**, **model experimentation**, and **web deployment** with **Streamlit**.

---

## 📌 Project Overview

- ✅ **Objective**: Predict the salary of tech job positions in the U.S. using only the job description text.
- 🔍 **Source**: Real job listings extracted from a custom job board via API + full descriptions scraped from individual pages.
- 🧠 **Approach**: Extract semantic features from job text using a **BERT-based NLP model**, train multiple ML models, and deploy a prediction web app.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn**
- **XGBoost**
- **Transformers (BERT)**
- **Pandas, NumPy**
- **BeautifulSoup + Requests**
- **Streamlit**
- **Render (for deployment)**

---

## 🔄 Workflow Summary

### 1. 🔗 Data Collection
- A custom **API** is used to fetch job listings (titles, locations, links).
- Then, **web scraping** is applied to retrieve full job descriptions from each job detail page.

### 2. 🌐 NLP & Feature Engineering
- A pre-trained **BERT** model is used to generate:
  - **Text embeddings**
  - **Named entities** (skills, technologies, seniority)
  - **Custom features** from job context

### 3. 📊 Exploratory Data Analysis (EDA)
- Analysis of salary distribution, correlations with extracted features, and outlier detection.

### 4. 🧪 Model Training
- Several regression models tested via **Scikit-learn**, including:
  - Linear Regression
  - Random Forest
  - Gradient Boosting

- Final optimization done using **XGBoost**, with hyperparameter tuning and cross-validation.

### 5. 🌍 Deployment
- An interactive web app built with **Streamlit** to:
  - Input a new job description
  - Predict the expected salary in real-time

- Deployed publicly using **Render**.

---

## 📸 App Demo (optional)

![App Screenshot](screenshot.png)  
📍 *Live demo:* [Insert Render URL here]

---

## ▶️ Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/salary-prediction-nlp.git
cd salary-prediction-nlp


## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## link de la aplicación 

https://final-project-kp5r.onrender.com/
