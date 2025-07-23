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

📍 *Live demo:* (https://final-project-kp5r.onrender.com/)

---


## 📁 Project Structure

The project is organized as follows:

salary-prediction-nlp/
├── src/
│ ├── app.py # Main Python script where the application runs
│ ├── explore.ipynb # Jupyter notebook for EDA and experimentation
│ └── utils.py # Helper functions (e.g., DB connections)
│
├── models/ # SQLAlchemy model classes (if using a DB)
│
├── data/ # Datasets organized by processing stage
│ ├── raw/ # Raw, unprocessed data
│ ├── interim/ # Temporarily transformed data
│ └── processed/ # Final, analysis-ready datasets
│
├── requirements.txt # List of required Python packages
└── README.md # Project documentation

