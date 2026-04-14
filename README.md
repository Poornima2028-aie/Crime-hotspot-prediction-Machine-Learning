# 🚔 Crime Hotspot Prediction using Spatio-Temporal Machine Learning

🚨 An intelligent machine learning system designed to analyze historical crime data and predict high-risk crime zones using spatial and temporal patterns.

---

## 📌 Overview
Crime analysis is a critical task for maintaining public safety, especially in rapidly growing urban environments. Traditional methods rely heavily on manual analysis and often fail to uncover hidden patterns in large datasets.

This project leverages **machine learning techniques** to:
- Analyze historical crime data  
- Identify spatial and temporal trends  
- Predict potential **crime hotspots**  

The system aims to support **proactive policing** by enabling authorities to take preventive measures rather than reacting after crimes occur.

---

## ❗ Problem Statement
Manual crime analysis is:
- Time-consuming  
- Inefficient for large datasets  
- Limited in detecting complex patterns  

There is a need for an automated system that can:
- Process large-scale crime data  
- Accurately identify high-risk areas  
- Assist in better decision-making and resource allocation  

---

## 🎯 Objectives
- Analyze crime datasets to understand patterns  
- Perform Exploratory Data Analysis (EDA)  
- Extract meaningful spatial and temporal features  
- Implement multiple machine learning models  
- Compare and evaluate model performance  

---

## 📊 Dataset Description
The dataset consists of crime records collected between **2020 and 2022**.

### Features include:
- Date and time of crime  
- Type of crime  
- Latitude and longitude  
- Community area  
- Arrest status  

These features help in capturing both **location-based (spatial)** and **time-based (temporal)** patterns.

---

## 🔍 Exploratory Data Analysis
EDA was performed to understand trends and patterns in the dataset.

### Key analyses include:
- Crime distribution by type  
- Crime count distribution  
- Crime occurrence by hour  
- Arrest vs non-arrest distribution  
- Crime density heatmap  

### Insights:
- Certain crimes occur more frequently  
- Crimes peak at specific hours  
- High-density regions indicate hotspot zones  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handled missing values  
- Converted date column to datetime format  
- Cleaned and filtered location data  

---

### 2. Feature Engineering
Extracted temporal features:
- Hour  
- Day of the week  
- Month  

Used spatial features:
- Latitude  
- Longitude  

---

### 3. Hotspot Label Creation
- Divided the geographic area into grid cells  
- Counted crime occurrences in each cell  
- Applied a threshold to classify:
  - Hotspot  
  - Non-hotspot  

---

### 4. Feature Selection
- Applied **ANOVA statistical test**  
- Selected the most relevant features for model training  

---

## 🤖 Machine Learning Models
The following models were implemented and evaluated:

- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

These models were chosen for their ability to handle structured data and capture complex patterns.

---

## 📈 Model Evaluation
Models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC Curve  

---

## 🏆 Results
- Gradient boosting models (**XGBoost** and **LightGBM**) showed the best performance  
- Models successfully captured spatial and temporal crime patterns  
- The system demonstrated strong predictive capability for identifying hotspots  

---

## 🚀 Applications
- Predictive policing  
- Crime prevention strategies  
- Resource allocation for law enforcement  
- Urban safety planning  

---

## 🔮 Future Work
- Integrate demographic and socio-economic data  
- Include weather data for enhanced predictions  
- Apply deep learning models  
- Deploy as a real-time web-based application  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM, CatBoost  

---

## 🧾 Conclusion
This project demonstrates how machine learning can be effectively used to analyze crime data and predict potential hotspot regions using spatial and temporal features.

By leveraging advanced models such as XGBoost and LightGBM, the system is capable of identifying high-risk areas with strong accuracy. This enables a shift from reactive to proactive crime management, helping law enforcement agencies allocate resources more efficiently and implement preventive strategies.

Overall, the project highlights the importance of data-driven approaches in enhancing public safety and supports the development of intelligent systems for real-world problem solving.
