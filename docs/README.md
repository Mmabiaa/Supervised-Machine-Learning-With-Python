# Supervised Machine Learning Assignment

Complete implementation of supervised machine learning workflows for both **regression** and **classification** tasks using Python and scikit-learn.

## 📁 Project Structure

```
├── customers.csv                      # Customer churn dataset
├── housing.csv                        # Housing price dataset
├── Supervised_ML_with Python.docx     # Assignment documentation
│
├── supervised_ml_complete.py          # ⭐ MAIN SCRIPT - Complete workflow
├── use_saved_models.py                # Demo: Using saved models
├── advanced_tuning.py                 # Advanced hyperparameter tuning
│
├── housing_price_model.joblib         # Saved regression model
├── housing_scaler.joblib              # Saved feature scaler
├── churn_prediction_model.joblib      # Saved classification model
│
├── housing_exploration.png            # Housing data visualizations
├── churn_model_results.png            # Churn model comparison
└── README.md                          # This file
```

## 🎯 Assignment Objectives

1. **Regression Task**: Predict housing prices based on features
2. **Classification Task**: Predict customer churn (will they leave?)

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib scipy
```

### Run the Complete Workflow

```bash
python supervised_ml_complete.py
```

This will:
- ✅ Load and explore both datasets
- ✅ Handle missing values and encode categories
- ✅ Train multiple models for both tasks
- ✅ Evaluate and compare models
- ✅ Save the best models
- ✅ Generate visualizations

## 📊 Part 1: Housing Price Prediction (Regression)

### Dataset: `housing.csv`
- **500 houses** with 5 features
- **Target**: `price` (continuous value)
- **Features**: sqft, bedrooms, age, neighborhood

### Models Trained:
1. **Linear Regression** - Simple baseline
2. **Ridge Regression** - With L2 regularization (BEST: R² = 0.9303)

### Key Results:
```
Best Model: Ridge Regression
R² Score:   0.9303
RMSE:       $18,818.52
MAE:        $15,508.42
```

### Usage Example:
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("housing_price_model.joblib")
scaler = joblib.load("housing_scaler.joblib")

# Prepare new data
new_house = pd.DataFrame({
    'sqft': [2000],
    'bedrooms': [3],
    'age': [10],
    'neighborhood_B': [1],
    'neighborhood_C': [0],
    'neighborhood_D': [0]
})

# Make prediction
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

## 🔄 Part 2: Customer Churn Prediction (Classification)

### Dataset: `customers.csv`
- **800 customers** with 6 features
- **Target**: `churn` (0 = stays, 1 = leaves)
- **Churn Rate**: 30.88%
- **Features**: age, income, tenure, contract_type, payment_method

### Models Trained:
1. **Logistic Regression** (BEST: F1 = 0.3636)
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**

### Key Results:
```
Best Model: Logistic Regression
Accuracy:   69.37%
Precision:  50.00%
Recall:     28.57%
F1-Score:   0.3636
ROC-AUC:    0.7384
```

### Confusion Matrix:
```
                Predicted
              Stay  Churn
Actual Stay    97    14
       Churn   35    14
```

### Usage Example:
```python
import joblib
import pandas as pd

# Load model (pipeline includes preprocessing!)
model = joblib.load("churn_prediction_model.joblib")

# Prepare new customer
new_customer = pd.DataFrame({
    'age': [45],
    'income': [60000],
    'tenure': [12],
    'contract_type': ['Month-to-month'],
    'payment_method': ['Electronic check']
})

# Make prediction
prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0]

print(f"Churn Prediction: {'YES' if prediction == 1 else 'NO'}")
print(f"Churn Probability: {probability[1]*100:.1f}%")
```

## 🔧 Advanced Features

### Hyperparameter Tuning

Run advanced hyperparameter optimization:

```bash
python advanced_tuning.py
```

This demonstrates:
- **Grid Search**: Exhaustive search over parameter grid
- **Randomized Search**: Efficient random sampling
- **Cross-Validation**: 5-fold CV for reliable estimates

## 📈 Key Techniques Demonstrated

### Data Preprocessing
- ✅ Handling missing values (imputation)
- ✅ Encoding categorical features (One-Hot Encoding)
- ✅ Feature scaling (StandardScaler)
- ✅ Train-test splitting (stratified for classification)

### Model Building
- ✅ Pipelines for reproducible workflows
- ✅ ColumnTransformer for mixed data types
- ✅ Multiple algorithm comparison
- ✅ Cross-validation for robust evaluation

### Evaluation Metrics
**Regression:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Classification:**
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Best Practices
- ✅ Never fit preprocessing on test data
- ✅ Use pipelines to prevent data leakage
- ✅ Cross-validation for model comparison
- ✅ Stratified splitting for imbalanced classes
- ✅ Save complete pipelines (preprocessing + model)

## 📝 Code Highlights

### Pipeline Example (Prevents Data Leakage!)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Everything in one object - fit once, use everywhere
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Fit on training data only
pipeline.fit(X_train, y_train)

# Automatically applies same preprocessing to test data
predictions = pipeline.predict(X_test)
```

### Handling Mixed Data Types
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["age", "income", "tenure"]),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop='first'))
    ]), ["contract_type", "payment_method"])
])
```

## 🎓 Learning Outcomes

By studying this code, you'll learn:

1. **Complete ML Workflow** - From data loading to model deployment
2. **Real-world Data Handling** - Missing values, mixed types, encoding
3. **Model Comparison** - Training and evaluating multiple algorithms
4. **Best Practices** - Pipelines, cross-validation, avoiding data leakage
5. **Model Persistence** - Saving and loading trained models
6. **Evaluation** - Choosing the right metrics for your problem

## 🔍 Understanding the Results

### Why is the Churn F1-Score Lower?

The churn prediction task is **harder** than housing price prediction because:

1. **Class Imbalance**: Only 31% of customers churn
2. **Complex Patterns**: Human behavior is less predictable than house prices
3. **Limited Features**: We only have 5 features to predict churn
4. **Real-world Challenge**: This reflects actual business problems!

**Improvements could include:**
- More features (customer complaints, service issues, competitor offers)
- Class balancing techniques (SMOTE, class weights)
- Ensemble methods (XGBoost, LightGBM)
- Feature engineering (customer lifetime value, usage patterns)

### Why is Housing R² So High?

The housing model performs well because:

1. **Strong Predictors**: Square footage highly correlates with price
2. **Clean Data**: Relatively few missing values
3. **Linear Relationships**: House prices follow predictable patterns
4. **Sufficient Data**: 500 samples is adequate for this problem

## 📚 Additional Resources

### Run Individual Scripts

**See all visualizations and detailed output:**
```bash
python supervised_ml_complete.py
```

**Test saved models on new data:**
```bash
python use_saved_models.py
```

**Optimize hyperparameters:**
```bash
python advanced_tuning.py
```

## 🏆 Key Takeaways

1. **Always split data before preprocessing** - Prevents leakage
2. **Use pipelines** - Makes code cleaner and prevents errors
3. **Cross-validate** - Single train-test split can be misleading
4. **Choose metrics wisely** - F1 for imbalanced classes, R² for regression
5. **Save complete pipelines** - Not just the model, but preprocessing too

## ✨ Summary

This assignment demonstrates a complete, professional-grade machine learning workflow:

- ✅ **2 complete ML projects** (regression + classification)
- ✅ **6+ algorithms** trained and compared
- ✅ **Proper validation** with cross-validation
- ✅ **Production-ready** with saved models and pipelines
- ✅ **Well-documented** with clear explanations

---

**Author**: Your Name  
**Course**: Data Science & Machine Learning  
**Date**: 2026

*All code follows scikit-learn best practices and the course guidelines.*
