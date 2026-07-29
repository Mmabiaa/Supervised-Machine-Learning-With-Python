"""
Supervised Machine Learning Complete Workflow
This script demonstrates both regression (housing prices) and classification (customer churn)
following the best practices from the course document.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("SUPERVISED MACHINE LEARNING WORKFLOW")
print("="*80)

# ============================================================================
# PART 1: REGRESSION - HOUSING PRICE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("PART 1: REGRESSION - Housing Price Prediction")
print("="*80)

# Load housing data
print("\n1. Loading housing data...")
housing_df = pd.read_csv("housing.csv")
print(f"✓ Loaded {housing_df.shape[0]} rows and {housing_df.shape[1]} columns")

# Explore the data
print("\n2. Exploring the data...")
print("\nFirst few rows:")
print(housing_df.head())
print("\nDataset info:")
print(housing_df.info())
print("\nStatistical summary:")
print(housing_df.describe())
print("\nMissing values:")
print(housing_df.isnull().sum())

# Visualize target distribution
print("\n3. Visualizing price distribution...")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(housing_df["price"], kde=True, bins=30)
plt.title("Housing Price Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
corr_matrix = housing_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("housing_exploration.png", dpi=100, bbox_inches='tight')
print("✓ Saved visualization to 'housing_exploration.png'")
plt.close()

# Handle missing values
print("\n4. Preprocessing - handling missing values...")
housing_df["bedrooms"] = housing_df["bedrooms"].fillna(housing_df["bedrooms"].median())
housing_df["neighborhood"] = housing_df["neighborhood"].fillna(housing_df["neighborhood"].mode()[0])
print(f"✓ Filled missing values - bedrooms: median, neighborhood: mode")

# Encode categorical features
print("\n5. Encoding categorical features...")
housing_encoded = pd.get_dummies(housing_df, columns=["neighborhood"], drop_first=True)
print(f"✓ One-hot encoded 'neighborhood' column")

# Split features and target
X_housing = housing_encoded.drop("price", axis=1)
y_housing = housing_encoded["price"]

# Split into train and test sets
print("\n6. Splitting data (80% train, 20% test)...")
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train_h.shape[0]} samples")
print(f"✓ Test set: {X_test_h.shape[0]} samples")

# Scale features
print("\n7. Scaling numeric features...")
scaler = StandardScaler()
X_train_h_scaled = scaler.fit_transform(X_train_h)
X_test_h_scaled = scaler.transform(X_test_h)
print("✓ Applied StandardScaler to all features")

# Train Linear Regression
print("\n8. Training Linear Regression model...")
lin_reg = LinearRegression()
lin_reg.fit(X_train_h_scaled, y_train_h)
y_pred_lr = lin_reg.predict(X_test_h_scaled)

mae_lr = mean_absolute_error(y_test_h, y_pred_lr)
mse_lr = mean_squared_error(y_test_h, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_h, y_pred_lr)

print(f"✓ Linear Regression Results:")
print(f"  - MAE:  ${mae_lr:,.2f}")
print(f"  - RMSE: ${rmse_lr:,.2f}")
print(f"  - R²:   {r2_lr:.4f}")

# Train Ridge Regression
print("\n9. Training Ridge Regression (with regularization)...")
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_h_scaled, y_train_h)
y_pred_ridge = ridge.predict(X_test_h_scaled)

mae_ridge = mean_absolute_error(y_test_h, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test_h, y_pred_ridge))
r2_ridge = r2_score(y_test_h, y_pred_ridge)

print(f"✓ Ridge Regression Results:")
print(f"  - MAE:  ${mae_ridge:,.2f}")
print(f"  - RMSE: ${rmse_ridge:,.2f}")
print(f"  - R²:   {r2_ridge:.4f}")

# Cross-validation
print("\n10. Performing 5-fold cross-validation...")
cv_scores = cross_val_score(
    LinearRegression(), X_train_h_scaled, y_train_h,
    cv=5, scoring='r2'
)
print(f"✓ Cross-validation R² scores: {cv_scores}")
print(f"✓ Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save the best model
print("\n11. Saving the best regression model...")
joblib.dump(ridge, "housing_price_model.joblib")
joblib.dump(scaler, "housing_scaler.joblib")
print("✓ Saved model to 'housing_price_model.joblib'")
print("✓ Saved scaler to 'housing_scaler.joblib'")

# ============================================================================
# PART 2: CLASSIFICATION - CUSTOMER CHURN PREDICTION
# ============================================================================

print("\n" + "="*80)
print("PART 2: CLASSIFICATION - Customer Churn Prediction")
print("="*80)

# Load churn data
print("\n1. Loading customer churn data...")
churn_df = pd.read_csv("customers.csv")
print(f"✓ Loaded {churn_df.shape[0]} rows and {churn_df.shape[1]} columns")

# Explore the data
print("\n2. Exploring the data...")
print("\nFirst few rows:")
print(churn_df.head())
print("\nChurn distribution:")
print(churn_df['churn'].value_counts())
print(f"\nChurn rate: {churn_df['churn'].mean()*100:.2f}%")
print("\nMissing values:")
print(churn_df.isnull().sum())

# Preprocessing for classification
print("\n3. Preprocessing - handling missing values...")
churn_df["income"] = churn_df["income"].fillna(churn_df["income"].median())
churn_df["contract_type"] = churn_df["contract_type"].fillna(churn_df["contract_type"].mode()[0])
print("✓ Filled missing values")

# Build a pipeline
print("\n4. Building preprocessing pipeline...")
numeric_features = ["age", "income", "tenure"]
categorical_features = ["contract_type", "payment_method"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop='first'))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("✓ Created preprocessing pipeline")

# Split data
print("\n5. Splitting data (stratified 80-20 split)...")
X_churn = churn_df.drop("churn", axis=1)
y_churn = churn_df["churn"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)
print(f"✓ Training set: {X_train_c.shape[0]} samples")
print(f"✓ Test set: {X_test_c.shape[0]} samples")
print(f"  Train churn rate: {y_train_c.mean()*100:.2f}%")
print(f"  Test churn rate: {y_test_c.mean()*100:.2f}%")

# Train multiple classification models
print("\n6. Training multiple classification models...")
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=6, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ]),
    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
}

results = {}
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    y_pred_proba = model.predict_proba(X_test_c)[:, 1] if hasattr(model.named_steps['classifier'], 'predict_proba') else None
    
    acc = accuracy_score(y_test_c, y_pred)
    prec = precision_score(y_test_c, y_pred)
    rec = recall_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'model': model,
        'predictions': y_pred
    }
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test_c, y_pred_proba)
        results[name]['auc'] = auc
    
    print(f"  ✓ {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    if y_pred_proba is not None:
        print(f"    ROC-AUC:   {auc:.4f}")

# Find best model
print("\n7. Identifying best model...")
best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
best_model = results[best_model_name]['model']
print(f"✓ Best model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

# Confusion matrix for best model
print("\n8. Confusion matrix for best model...")
cm = confusion_matrix(y_test_c, results[best_model_name]['predictions'])
print(cm)
print("\nClassification Report:")
print(classification_report(y_test_c, results[best_model_name]['predictions']))

# Visualize results
print("\n9. Creating visualizations...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
model_names = list(results.keys())
f1_scores = [results[name]['f1'] for name in model_names]
colors = ['lightcoral' if name != best_model_name else 'lightgreen' for name in model_names]
plt.barh(model_names, f1_scores, color=colors)
plt.xlabel('F1-Score')
plt.title('Model Comparison - F1 Scores')
plt.xlim(0, 1)
for i, v in enumerate(f1_scores):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')

plt.tight_layout()
plt.savefig("churn_model_results.png", dpi=100, bbox_inches='tight')
print("✓ Saved visualization to 'churn_model_results.png'")
plt.close()

# Cross-validation on best model
print("\n10. Performing 5-fold cross-validation on best model...")
cv_scores_churn = cross_val_score(
    best_model, X_train_c, y_train_c, cv=5, scoring='f1'
)
print(f"✓ Cross-validation F1 scores: {cv_scores_churn}")
print(f"✓ Mean F1: {cv_scores_churn.mean():.4f} (+/- {cv_scores_churn.std():.4f})")

# Save the best classification model
print("\n11. Saving the best classification model...")
joblib.dump(best_model, "churn_prediction_model.joblib")
print("✓ Saved model to 'churn_prediction_model.joblib'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WORKFLOW COMPLETE - SUMMARY")
print("="*80)

print("\n📊 REGRESSION (Housing Prices):")
print(f"   Best Model: Ridge Regression")
print(f"   R² Score: {r2_ridge:.4f}")
print(f"   RMSE: ${rmse_ridge:,.2f}")
print(f"   Files: housing_price_model.joblib, housing_scaler.joblib")

print("\n📊 CLASSIFICATION (Customer Churn):")
print(f"   Best Model: {best_model_name}")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   File: churn_prediction_model.joblib")

print("\n✅ All models trained, evaluated, and saved successfully!")
print("="*80)
