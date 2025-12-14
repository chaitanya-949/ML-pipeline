# 📉 Churn Analysis with DVC Experiment Tracking

## Overview

This project focuses on **Churn Analysis** using a complete **end-to-end Machine Learning pipeline**. The pipeline is modular, reproducible, and production-oriented, with **DVC (Data Version Control)** used for **data versioning and experiment tracking**.

The objective is to predict customer churn by processing raw data, applying NLP-based feature engineering, training multiple models, and evaluating them while tracking experiments in a systematic manner.

---

## Key Features

* End-to-end ML pipeline (Ingestion → Preprocessing → Feature Engineering → Modeling → Evaluation)
* Modular and reusable Python scripts
* Experiment tracking using **DVC + dvclive**
* Parameter management via `params.yaml`
* Extensive logging for debugging and traceability
* Model comparison (RandomForest & XGBoost)

---

## Project Structure

```
├── data/
│   ├── raw/                # Train/Test raw data
│   ├── interim/            # Preprocessed text data
│   └── processed/          # TF-IDF transformed data
│
├── logs/                   # Pipeline execution logs
├── models/                 # Trained model artifacts
├── reports/                # Evaluation metrics (JSON)
│
├── data_ingestion.py
├── data_preprocessing.py
├── feature_engineering.py
├── model_building.py
├── model_evaluation.py
│
├── params.yaml              # Centralized configuration
├── dvc.yaml                 # DVC pipeline definition
├── README.md
└── requirements.txt
```

---

## Pipeline Stages

### 1. Data Ingestion

* Loads raw dataset
* Performs initial cleaning and column renaming
* Splits data into train and test sets
* Stores versioned data using DVC

**Script:** `data_ingestion.py`

---

### 2. Data Preprocessing

* Label encoding of target variable
* Duplicate removal
* Text normalization:

  * Lowercasing
  * Tokenization
  * Stopword removal
  * Stemming

**Script:** `data_preprocessing.py`

---

### 3. Feature Engineering

* TF-IDF vectorization on processed text data
* Configurable number of features via `params.yaml`
* Outputs numerical feature matrices

**Script:** `feature_engineering.py`

---

### 4. Model Building

* Trains multiple models:

  * Random Forest Classifier
  * XGBoost Classifier
* Saves trained models as serialized artifacts

**Script:** `model_building.py`

---

### 5. Model Evaluation & Experiment Tracking

* Evaluates models using:

  * Accuracy
  * Precision
  * Recall
  * ROC-AUC
* Logs metrics and parameters using **dvclive**
* Tracks experiments using **DVC**

**Script:** `model_evaluation.py`

---

## Experiment Tracking with DVC

* Data, models, and metrics are versioned
* Each experiment is reproducible
* Compare multiple runs using:

```bash
dvc exp show
```

Metrics and parameters are automatically logged for every experiment run.

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
```

### 3. Run the Complete Pipeline

```bash
dvc repro
```

---

## Configuration

All tunable parameters are managed via `params.yaml`, including:

* Train-test split ratio
* TF-IDF max features
* Model hyperparameters

This enables easy experimentation without changing code.

---

## Results

* Evaluation metrics are stored in the `reports/` directory
* Models are saved in the `models/` directory
* Experiments can be compared using DVC CLI

---

## Tools & Technologies

* **Python**
* **Scikit-learn**
* **XGBoost**
* **NLTK**
* **DVC & dvclive**
* **YAML**

---

## Author

**Chaitanya**
Machine Learning / Data Science Practitioner

---

## Future Improvements

* Hyperparameter tuning with DVC experiments
* Model explainability (SHAP / LIME)
* CI/CD integration for ML pipelines
* Deployment using FastAPI or Streamlit

---

⭐ If you find this project useful, feel free to star the repository!
