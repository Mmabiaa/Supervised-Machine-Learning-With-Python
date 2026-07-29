# Supervised Machine Learning Assignment
## Complete Implementation with Regression and Classification

**Course:** Data Science & Machine Learning
**Date:** July 29, 2026
**Author:** Boateng Prince Agyenim

---

## 📋 Project Overview

This project demonstrates professional-grade supervised machine learning workflows for:
1. **Regression**: Housing price prediction
2. **Classification**: Customer churn prediction

The implementation follows industry best practices including proper data preprocessing, pipeline usage, cross-validation, and model deployment.

---

## 📁 Project Structure

```
Project/
│
├── data/                          # Raw datasets
│   ├── housing.csv               # Housing price data (500 samples)
│   └── customers.csv             # Customer churn data (800 samples)
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Regression_Housing_Prices.ipynb
│   └── 03_Classification_Customer_Churn.ipynb
│
├── src/                           # Python scripts
│   ├── supervised_ml_complete.py       # Main workflow script
│   ├── interactive_exploration.py      # Data exploration
│   ├── use_saved_models.py            # Model deployment demo
│   └── advanced_tuning.py             # Hyperparameter optimization
│
├── models/                        # Trained models
│   ├── housing_price_model.joblib
│   ├── housing_scaler.joblib
│   └── churn_prediction_model.joblib
│
├── visualizations/                # Generated plots
│   ├── housing_exploration.png
│   ├── churn_model_results.png
│   ├── model_comparison_regression.png
│   ├── model_comparison_classification.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
├── docs/                          # Documentation
│   ├── README.md                 # This file
│   ├── QUICK_START.md           # Quick start guide
│   └── ASSIGNMENT_SUMMARY.md    # Detailed results report
│
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebooks (Recommended)

```bash
jupyter notebook
```

Navigate to `notebooks/` and run in order:
1. `01_Data_Exploration.ipynb` - Understand the data
2. `02_Regression_Housing_Prices.ipynb` - Housing price prediction
3. `03_Classification_Customer_Churn.ipynb` - Churn prediction

### 3. Or Run Python Scripts

```bash
# Complete workflow (both regression and classification)
python src/supervised_ml_complete.py

# Interactive data exploration
python src/interactive_exploration.py

# Test saved models
python src/use_saved_models.py

# Advanced hyperparameter tuning
python src/advanced_tuning.py
```

---

## 📊 Results Summary

### Housing Price Prediction (Regression)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.9281 | $19,114 | $15,675 |
| **Ridge (Best)** | **0.9303** | **$18,819** | **$15,508** |
| Lasso | 0.9295 | $18,925 | $15,562 |
| ElasticNet | 0.9294 | $18,930 | $15,568 |

**Best Model:** Ridge Regression  
**Performance:** Excellent (93% variance explained)  
**Cross-Validation:** Mean R² = 0.9128 (±0.0067)

### Customer Churn Prediction (Classification)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression (Best)** | **0.694** | **0.500** | **0.286** | **0.364** | **0.738** |
| Random Forest | 0.681 | 0.464 | 0.265 | 0.338 | 0.687 |
| Gradient Boosting | 0.675 | 0.452 | 0.265 | 0.333 | 0.702 |
| Decision Tree | 0.656 | 0.412 | 0.286 | 0.337 | 0.647 |
| KNN | 0.644 | 0.382 | 0.265 | 0.313 | 0.618 |

**Best Model:** Logistic Regression  
**Performance:** Moderate (typical for churn prediction)  
**Cross-Validation:** Mean F1 = 0.4312 (±0.0428)

---

## 🎯 Key Features

### Technical Implementation
✅ Complete preprocessing pipelines  
✅ Proper train-test splitting (no data leakage)  
✅ Stratified splitting for imbalanced classes  
✅ Feature scaling and encoding  
✅ Cross-validation for robust evaluation  
✅ Hyperparameter tuning examples  
✅ Model persistence (saving/loading)  
✅ Production-ready code  

### Best Practices
✅ Pipelines prevent data leakage  
✅ Appropriate metrics for each task  
✅ Cross-validation for reliability  
✅ Comprehensive visualizations  
✅ Well-documented code  
✅ Modular, reusable structure  

---

## 📚 Documentation
- **Jupyter Notebooks** - Interactive, step-by-step guides in `notebooks/`

---

## 🛠️ Technologies Used

- **Python 3.12**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning models and tools
- **matplotlib & seaborn** - Visualizations
- **joblib** - Model persistence
- **jupyter** - Interactive notebooks

---

## 📈 Visualizations

All visualizations are saved in the `visualizations/` folder:

1. **housing_exploration.png** - Price distribution and correlations
2. **churn_model_results.png** - Model comparison for churn
3. **model_comparison_regression.png** - Regression model metrics
4. **model_comparison_classification.png** - Classification model metrics
5. **confusion_matrix.png** - Churn prediction confusion matrix
6. **roc_curves.png** - ROC curves for all classification models

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Complete ML Workflow** - From data loading to deployment
2. **Regression & Classification** - Both major supervised learning tasks
3. **Data Preprocessing** - Missing values, encoding, scaling
4. **Model Selection** - Training and comparing multiple algorithms
5. **Evaluation** - Appropriate metrics for different problems
6. **Best Practices** - Pipelines, CV, avoiding data leakage
7. **Production Deployment** - Model saving and loading

---

## 📝 Key Insights

### Housing Price Model
- **Strong Performance**: 93% R² indicates excellent predictions
- **Key Predictor**: Square footage is most important (r=0.908)
- **Neighborhood Effect**: Up to $56K price difference between areas
- **Model Choice**: Ridge regularization improves generalization

### Churn Prediction Model
- **Moderate Performance**: 69% accuracy, 74% AUC (typical for churn)
- **Class Imbalance**: Only 31% churn rate affects recall
- **Business Value**: Identifies some high-risk customers for retention
- **Contract Insight**: Month-to-month contracts have 42% churn vs 5% for two-year

---

## 🔧 Customization

To adapt this project for your own data:

1. Replace CSV files in `data/` folder
2. Update column names in notebooks/scripts
3. Adjust preprocessing steps as needed
4. Re-run the workflow

---

## 📦 Deliverables

For submission, this project includes:

1. ✅ **3 Jupyter Notebooks** - Interactive analysis
2. ✅ **4 Python Scripts** - Automated workflows
3. ✅ **3 Trained Models** - Ready for deployment
4. ✅ **6 Visualizations** - Results and insights
5. ✅ **Complete Documentation** - README, guides, summary

---

## 🚦 Reproducibility

All random states are fixed (`random_state=42`) for reproducible results:
- Train-test splits
- Model training
- Cross-validation folds

---

## 💡 Future Improvements

### Housing Model
- Add more features (school districts, crime rates, amenities)
- Try polynomial features for non-linear relationships
- Ensemble methods (stacking, blending)

### Churn Model
- Collect more features (usage patterns, complaints, satisfaction scores)
- Address class imbalance (SMOTE, class weights)
- Try advanced models (XGBoost, Neural Networks)
- Cost-sensitive learning (weight false negatives higher)

---

## 📞 Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the Jupyter notebooks for examples
3. Examine code comments in `src/` scripts

---

## ✅ Assignment Checklist

- [x] Data exploration and visualization
- [x] Proper preprocessing (missing values, encoding, scaling)
- [x] Train-test splitting with no data leakage
- [x] Multiple regression models trained and evaluated
- [x] Multiple classification models trained and evaluated
- [x] Appropriate metrics used for each task
- [x] Cross-validation performed
- [x] Models saved for deployment
- [x] Comprehensive visualizations
- [x] Well-documented code
- [x] Professional project structure
- [x] Reproducible results

---

**Project Status:** Complete and Ready for Submission

**Execution Time:** ~1 minute for complete workflow  
**Lines of Code:** 800+ lines across all scripts  
**Models Trained:** 9 algorithms (4 regression + 5 classification)  
**Documentation:** 4 comprehensive documents  

---

*This project follows best practices from the course "Supervised ML with Python" and scikit-learn documentation.*
