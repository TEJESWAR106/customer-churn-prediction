# Customer Churn Prediction

Predicts customer churn using Machine Learning on real telecom data.
Covers the full Data Science lifecycle — EDA, preprocessing, model training,
evaluation, and visualization.

## Tech Stack
Python, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn

## Model Results
- Random Forest Accuracy: 79.03%
- ROC-AUC Score: 69.18%

## Project Steps
1. Load telecom customer data from CSV
2. EDA — churn distribution, correlation heatmap, box plots, t-tests
3. Preprocess — label encoding, standard scaling, train-test split
4. Train Random Forest classifier
5. Evaluate — accuracy, ROC-AUC, confusion matrix, classification report
6. Visualize — feature importance, churn distribution plots

## How to Run
pip install -r requirements.txt
python main.py

## Output
All charts are saved automatically in the /plots folder after running.