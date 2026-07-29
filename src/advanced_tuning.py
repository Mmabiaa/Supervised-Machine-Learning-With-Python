"""
Advanced Hyperparameter Tuning Example
Demonstrates Grid Search and Random Search for optimal model parameters.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
import time

print("="*80)
print("ADVANCED HYPERPARAMETER TUNING")
print("="*80)

# Load and prepare the churn data
print("\n1. Loading and preparing customer churn data...")
churn_df = pd.read_csv("customers.csv")
churn_df["income"] = churn_df["income"].fillna(churn_df["income"].median())
churn_df["contract_type"] = churn_df["contract_type"].fillna(churn_df["contract_type"].mode()[0])

X = churn_df.drop("churn", axis=1)
y = churn_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Create preprocessing pipeline
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

# ============================================================================
# GRID SEARCH - Random Forest
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH - Random Forest")
print("="*80)

print("\n2. Setting up Grid Search...")
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [4, 6, 8, 10],
    "classifier__min_samples_split": [2, 5, 10],
}

print(f"   Parameter combinations to test: {3 * 4 * 3} = 36")

print("\n3. Running Grid Search with 5-fold CV...")
start_time = time.time()

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
elapsed_time = time.time() - start_time

print(f"\n✓ Grid Search completed in {elapsed_time:.2f} seconds")
print(f"\n📊 GRID SEARCH RESULTS:")
print(f"   Best F1-Score: {grid_search.best_score_:.4f}")
print(f"   Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"      {param}: {value}")

# Test on test set
test_score = grid_search.score(X_test, y_test)
print(f"   Test Set F1-Score: {test_score:.4f}")

# Show top 5 parameter combinations
print("\n   Top 5 Parameter Combinations:")
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
for idx, row in results_df.head(5).iterrows():
    print(f"      Rank {int(row['rank_test_score'])}: F1={row['mean_test_score']:.4f} "
          f"(n_est={row['param_classifier__n_estimators']}, "
          f"max_d={row['param_classifier__max_depth']}, "
          f"min_samp={row['param_classifier__min_samples_split']})")

# ============================================================================
# RANDOMIZED SEARCH - Gradient Boosting
# ============================================================================

print("\n" + "="*80)
print("RANDOMIZED SEARCH - Gradient Boosting")
print("="*80)

print("\n4. Setting up Randomized Search...")
gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

# Define parameter distributions
param_distributions = {
    "classifier__n_estimators": randint(50, 300),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(3, 10),
    "classifier__min_samples_split": randint(2, 20),
    "classifier__subsample": uniform(0.6, 0.4),
}

print(f"   Random combinations to test: 30")

print("\n5. Running Randomized Search with 5-fold CV...")
start_time = time.time()

random_search = RandomizedSearchCV(
    gb_pipeline,
    param_distributions,
    n_iter=30,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)
elapsed_time = time.time() - start_time

print(f"\n✓ Randomized Search completed in {elapsed_time:.2f} seconds")
print(f"\n📊 RANDOMIZED SEARCH RESULTS:")
print(f"   Best F1-Score: {random_search.best_score_:.4f}")
print(f"   Best Parameters:")
for param, value in random_search.best_params_.items():
    if 'learning_rate' in param or 'subsample' in param:
        print(f"      {param}: {value:.4f}")
    else:
        print(f"      {param}: {value}")

# Test on test set
test_score = random_search.score(X_test, y_test)
print(f"   Test Set F1-Score: {test_score:.4f}")

# Show top 5 parameter combinations
print("\n   Top 5 Parameter Combinations:")
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
for idx, row in results_df.head(5).iterrows():
    print(f"      Rank {int(row['rank_test_score'])}: F1={row['mean_test_score']:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: Grid Search vs Randomized Search")
print("="*80)

print(f"\n{'Method':<20} {'Best CV F1':<15} {'Test F1':<15}")
print("-" * 50)
print(f"{'Grid Search (RF)':<20} {grid_search.best_score_:<15.4f} Not computed")
print(f"{'Random Search (GB)':<20} {random_search.best_score_:<15.4f} Not computed")

print("\n💡 KEY INSIGHTS:")
print("   • Grid Search: Exhaustive but slow - tests all combinations")
print("   • Random Search: Fast and often finds near-optimal solutions")
print("   • Use Grid Search for small parameter spaces")
print("   • Use Random Search for large parameter spaces")
print("   • Always validate with cross-validation")

print("\n✅ Hyperparameter tuning complete!")
print("="*80)
