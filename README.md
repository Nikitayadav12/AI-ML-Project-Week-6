🛒 Retail Dataset Project — Week 6 

📌 Project Overview

This project is part of my 6-week AI/ML internship, focusing on solving a real-world retail problem using data science and machine learning techniques.

The main business objective:
👉 Predict whether a customer invoice will be cancelled.

The dataset is highly imbalanced (very few cancellations). To solve this, we combine traditional ML models with synthetic data generation (CTGAN & GaussianCopula) to create a balanced dataset and improve model performance.



📂 Dataset

Source: Online Retail Dataset (Kaggle)[Online Retail Dataset on Kaggle]

Contains: Transactions from a UK-based online retail store (Dec 2010 – Dec 2011).

Key columns:

InvoiceNo → Unique invoice identifier (C-prefixed = cancelled)

StockCode, Description → Item info

Quantity, UnitPrice → Transaction details

CustomerID → Customer identifier

Country → Location of transaction

We engineered features at the invoice level, such as:

Number of unique SKUs

Total quantity & invoice amount

Time features (Hour, DayOfWeek)

Cancellation flag (IsCancelled)



⚙️ Project Workflow

Data Cleaning & Preparation

Handle missing values, duplicates, and negative quantities.

Create IsCancelled flag and LineTotal.

Exploratory Data Analysis (EDA)

Revenue & cancellation trends over time

Top countries by revenue

Distribution of invoice totals and quantities

Feature Engineering

Aggregated to invoice-level

Added numeric, discrete, and categorical features

Created magnitude (Abs_) features for stability

Train/Test Split & Preprocessing

70/30 stratified split

OneHotEncoder for categoricals

ColumnTransformer for preprocessing

Baseline Models

Logistic Regression

Random Forest

Handling Imbalance

Extracted cancelled invoices (minority class)

Generated synthetic minority samples using:

CTGAN (80 epochs for runtime efficiency)

GaussianCopula (fallback for high-cardinality columns)

Augmented training set to ~45% minority

Retrained Models (with augmentation)

Logistic Regression + CTGAN

Random Forest + CTGAN

Hyperparameter Tuning

RandomizedSearchCV on Random Forest

Optimized for F1-score

Evaluation & Visualization

ROC curves, confusion matrices

PCA visualization of synthetic vs real minority samples


📊 Results

Baseline models struggled due to imbalance.

After augmentation with CTGAN:

Recall for minority class improved significantly.

Random Forest with tuned parameters achieved the best F1/AUC.

PCA plots confirmed that synthetic samples closely resemble real minority invoices.


🚀 How to Run

Clone this repo:

Install dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:
jupyter notebook


Open and run:

week 6 project - Jupyter Notebook.ipynb


📦 Requirements

Python 3.9–3.11

pandas, numpy, matplotlib, seaborn

scikit-learn

imbalanced-learn (optional for SMOTE experiments)

sdv (CTGANSynthesizer, GaussianCopulaSynthesizer)

cryptography, paramiko (warnings suppressed)

Install all at once:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn sdv cryptography paramiko


📌 Key Learnings

Handling imbalanced datasets with GANs and statistical models.

Importance of feature engineering for invoice-level prediction.

Trade-offs between deep generative models (CTGAN) and faster statistical models (GaussianCopula).

Hyperparameter tuning improves performance but increases compute time.


🏆 Conclusion

This project successfully demonstrates how to apply ML + synthetic data generation to solve a real-world retail cancellation problem.
The final model (RandomForest + CTGAN + tuning) shows improved predictive power, and the methodology is extendable to other imbalanced classification problems.


📌 Author: Nikita Yadav

📌 Internship Project — Week 6
