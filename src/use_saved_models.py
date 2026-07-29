"""
Demonstration: How to Load and Use Saved Models
This script shows how to load the trained models and make predictions on new data.
"""

import numpy as np
import pandas as pd
import joblib

print("="*80)
print("USING SAVED SUPERVISED ML MODELS")
print("="*80)

# ============================================================================
# PART 1: Using the Housing Price Prediction Model
# ============================================================================

print("\n" + "="*80)
print("PART 1: Housing Price Prediction")
print("="*80)

# Load the saved model and scaler
print("\n1. Loading saved housing model and scaler...")
housing_model = joblib.load("housing_price_model.joblib")
housing_scaler = joblib.load("housing_scaler.joblib")
print("✓ Model and scaler loaded successfully")

# Create sample new data
print("\n2. Creating sample new house data...")
new_houses = pd.DataFrame({
    'sqft': [2000, 1500, 2500],
    'bedrooms': [3, 2, 4],
    'age': [10, 25, 5],
    'neighborhood_B': [1, 0, 0],  # One-hot encoded
    'neighborhood_C': [0, 1, 0],
    'neighborhood_D': [0, 0, 1]
})

print("\nNew houses to predict:")
print(new_houses)

# Scale the features
print("\n3. Scaling features...")
new_houses_scaled = housing_scaler.transform(new_houses)
print("✓ Features scaled")

# Make predictions
print("\n4. Making price predictions...")
predicted_prices = housing_model.predict(new_houses_scaled)

print("\n📈 PREDICTED PRICES:")
for i, price in enumerate(predicted_prices):
    print(f"   House {i+1}: ${price:,.2f}")

# ============================================================================
# PART 2: Using the Customer Churn Prediction Model
# ============================================================================

print("\n" + "="*80)
print("PART 2: Customer Churn Prediction")
print("="*80)

# Load the saved model (pipeline includes preprocessing)
print("\n1. Loading saved churn prediction model...")
churn_model = joblib.load("churn_prediction_model.joblib")
print("✓ Model loaded successfully")

# Create sample new customer data
print("\n2. Creating sample new customer data...")
new_customers = pd.DataFrame({
    'age': [45, 25, 65],
    'income': [60000, 35000, 80000],
    'tenure': [12, 2, 48],
    'contract_type': ['Month-to-month', 'Two year', 'One year'],
    'payment_method': ['Electronic check', 'Bank transfer', 'Credit card']
})

print("\nNew customers to predict:")
print(new_customers)

# Make predictions (pipeline handles all preprocessing)
print("\n3. Making churn predictions...")
churn_predictions = churn_model.predict(new_customers)
churn_probabilities = churn_model.predict_proba(new_customers)

print("\n📊 CHURN PREDICTIONS:")
for i, (pred, proba) in enumerate(zip(churn_predictions, churn_probabilities)):
    churn_status = "WILL CHURN" if pred == 1 else "WILL STAY"
    churn_prob = proba[1] * 100
    print(f"   Customer {i+1}: {churn_status} (Churn probability: {churn_prob:.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ Successfully loaded and used both saved models!")
print("="*80)
print("\n💡 Key Takeaways:")
print("   - Saved models can be loaded with joblib.load()")
print("   - Preprocessing steps are preserved in pipelines")
print("   - New data must match the training data format")
print("   - Scalers must be applied in the same order as training")
print("="*80)
