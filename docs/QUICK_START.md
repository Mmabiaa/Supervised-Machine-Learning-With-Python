# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Explore the Data (5 minutes)
```bash
python interactive_exploration.py
```
**What it does**: 
- Shows you both datasets
- Creates visualizations
- Reveals key patterns and insights

**Output**: 
- `interactive_housing_analysis.png`
- `interactive_churn_analysis.png`

---

### Step 2: Train All Models (1 minute)
```bash
python supervised_ml_complete.py
```
**What it does**:
- Trains 2 regression models (housing prices)
- Trains 4 classification models (customer churn)
- Evaluates and compares all models
- Saves the best models

**Output**:
- `housing_price_model.joblib` (R² = 0.93)
- `churn_prediction_model.joblib` (F1 = 0.36)
- `housing_exploration.png`
- `churn_model_results.png`

---

### Step 3: Use the Models (2 minutes)
```bash
python use_saved_models.py
```
**What it does**:
- Loads saved models
- Makes predictions on new data
- Shows you how to deploy models

---

## 📊 What You Get

### 🏠 Housing Price Predictor
- **Accuracy**: 93% (R² score)
- **Error**: ±$18,818 average
- **Use**: Predict prices for new houses

**Example**:
```python
import joblib
model = joblib.load("housing_price_model.joblib")
# Predict: 2000 sqft, 3 bed, 10 years old, neighborhood B
# Result: $321,579
```

### 👥 Churn Predictor
- **Accuracy**: 69% overall
- **Catches**: 29% of churners
- **Use**: Identify at-risk customers

**Example**:
```python
import joblib
model = joblib.load("churn_prediction_model.joblib")
# Predict: 45 yrs, $60K income, 12 months, month-to-month
# Result: 58.8% churn probability
```

---

## 🎯 Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `interactive_exploration.py` | Learn about the data | Start here! |
| `supervised_ml_complete.py` | Train all models | Main workflow |
| `use_saved_models.py` | Deploy models | After training |
| `advanced_tuning.py` | Optimize parameters | For better results |
| `README.md` | Full documentation | Reference guide |
| `ASSIGNMENT_SUMMARY.md` | Results report | Show your work |

---

## 💡 Common Questions

**Q: Which script should I run first?**  
A: Run `interactive_exploration.py` to understand the data, then `supervised_ml_complete.py` to train models.

**Q: How do I make predictions on my own data?**  
A: See `use_saved_models.py` for examples. Load the model with `joblib.load()`, then call `.predict()`.

**Q: Can I improve the models?**  
A: Yes! Run `advanced_tuning.py` to try different hyperparameters, or modify `supervised_ml_complete.py` to test other algorithms.

**Q: Why is the churn model accuracy lower?**  
A: Predicting human behavior (churn) is harder than house prices. 69% accuracy and 74% AUC are reasonable for this problem.

**Q: What if I get import errors?**  
A: Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn joblib scipy`

---

## 📈 Expected Results

### Housing Model
- **Training Time**: ~2 seconds
- **R² Score**: 0.93 (93% variance explained)
- **Prediction Range**: $96K - $489K

### Churn Model
- **Training Time**: ~5 seconds
- **F1 Score**: 0.36
- **ROC-AUC**: 0.74
- **Churn Rate**: 31%

---

## 🎓 What You'll Learn

By running these scripts, you'll see:

1. ✅ How to handle missing data
2. ✅ How to encode categories (text → numbers)
3. ✅ How to scale features properly
4. ✅ How to split data (train/test)
5. ✅ How to train multiple models
6. ✅ How to evaluate models correctly
7. ✅ How to save and load models
8. ✅ How to make predictions on new data

---

## 🔥 Pro Tips

**Tip 1**: Always run `interactive_exploration.py` first to understand your data.

**Tip 2**: The housing model is better because it has stronger patterns. The churn model is realistic - predicting customer behavior is hard!

**Tip 3**: Use pipelines (already implemented) to avoid data leakage - a common mistake that makes models look better than they are.

**Tip 4**: Cross-validation (5-fold) gives you more reliable scores than a single train-test split.

**Tip 5**: For churn, F1-score matters more than accuracy because classes are imbalanced (69% stay, 31% churn).

---

## 🎬 Complete Workflow (10 minutes)

```bash
# 1. Explore data (5 min)
python interactive_exploration.py

# 2. Train models (1 min)
python supervised_ml_complete.py

# 3. Test predictions (2 min)
python use_saved_models.py

# 4. Optimize (optional, 2 min)
python advanced_tuning.py
```

---

## ✅ Success Checklist

- [ ] All scripts run without errors
- [ ] 3 model files saved (.joblib)
- [ ] 4 visualization files created (.png)
- [ ] Housing R² > 0.90
- [ ] Churn F1 > 0.30
- [ ] Understand both datasets
- [ ] Can explain model results

---

## 🆘 Troubleshooting

**Error: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**Error: "File not found: customers.csv"**
- Make sure you're in the correct directory
- Check that all CSV files are present

**Models perform differently**
- This is normal! Different train-test splits give slightly different results
- The random_state=42 should make results reproducible

**Warnings about convergence**
- Increase max_iter in LogisticRegression
- These warnings are usually harmless

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Look at code comments in each script
3. Review `ASSIGNMENT_SUMMARY.md` for expected results

---

**Ready?** Start with: `python interactive_exploration.py` 🚀
