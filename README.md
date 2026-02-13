# Clinical Risk Assessment Module (CRAM) 🩺
**Internal Diagnostic Screening Tool | Protocol v4.2.1**

A high-fidelity Clinical Decision Support System (CDSS) for diabetes risk stratification. This module utilizes a Histogram-based Gradient Boosting engine to process patient metrics and generate standardized clinical risk reports.

## 🚀 Technical Architecture
This application is built with a **Strict No-JS Architecture**, utilizing server-side Python rendering (Flask/Jinja2) to ensure maximum security, stability, and compatibility across professional clinical hardware.



### Core Features
* **ML Engine:** Histogram-based Gradient Boosting Classifier (Scikit-Learn).
* **High-Density UI:** EHR-inspired interface designed for rapid data entry and professional clarity.
* **Predictive Metrics:** Full integration of Age, BMI, HbA1c, Blood Glucose, Hypertension, Heart Disease, and Gender.
* **Server-Side Logic:** 0% Client-side JavaScript. All risk calculations and report generation occur on the secure Python backend to protect model integrity.

### 📂 Project Structure
* **app.py**: Central Flask server containing preprocessing pipelines and ML prediction logic.
* **diabetes_dataset.csv**: Training data source for the HistGradientBoosting model.
* **templates/index.html**: The clinical interface, styled with hospital-standard CSS and rendered via Jinja2.
* **requirements.txt**: Configuration for cloud deployment environments.

---
**Disclaimer:** *FOR SCREENING USE ONLY. NOT A SUBSTITUTE FOR CLINICAL DIAGNOSIS. All data processing follows internal protocol v4.2.1 regarding patient data privacy.*
