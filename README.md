# Customer Churn Prediction using Machine Learning

## Project Overview
Customer churn is a major challenge for businesses, especially in the telecommunications sector.  
This project focuses on building a **machine learning model** to predict whether a customer is likely to churn based on historical customer data.

The goal is to help businesses **identify at-risk customers early** and take proactive retention measures.

---

## Dataset
- The dataset contains customer demographic details, service usage information, and billing data.
- Target variable: **Churn** (Yes / No)
- Dataset type: Structured tabular data (CSV)

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

---

## Machine Learning Workflow
1. Data Loading and Cleaning  
2. Handling Missing Values  
3. Encoding Categorical Variables  
4. Train–Test Split  
5. Model Training  
6. Model Evaluation  

---

## Model Used
- **Random Forest Classifier**(Binary Classification)

---

## Model Evaluation
The model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

These metrics help assess how well the model predicts customer churn and balances false positives and false negatives.

---

## Results
The trained model achieved reliable performance on unseen test data, demonstrating its ability to predict customer churn effectively.

(**Exact metric values can be added here if required.**)

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Feature selection optimization
- Handling class imbalance
- Deployment using Streamlit or Flask

---

## 📎 How to Run the Project
```bash
pip install -r requirements.txt