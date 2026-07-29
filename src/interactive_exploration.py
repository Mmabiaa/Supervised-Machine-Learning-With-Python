"""
Interactive Exploration Script
Step-by-step guide to understand the datasets and models
Perfect for learning and experimenting!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("INTERACTIVE SUPERVISED ML EXPLORATION")
print("="*80)
print("\n👋 This script walks you through the data step-by-step")
print("   You can modify and re-run sections to experiment!\n")

# ============================================================================
# SECTION 1: Housing Data Exploration
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Exploring the Housing Dataset")
print("="*80)

# Load the data
housing_df = pd.read_csv("housing.csv")

print("\n📊 Dataset Overview:")
print(f"   Rows: {housing_df.shape[0]}")
print(f"   Columns: {housing_df.shape[1]}")
print(f"   Features: {list(housing_df.columns)}")

print("\n📝 First 10 rows:")
print(housing_df.head(10))

print("\n📈 Statistical Summary:")
print(housing_df.describe())

print("\n❓ Missing Values:")
missing = housing_df.isnull().sum()
print(missing[missing > 0])

print("\n🏠 Questions to explore:")
print("   1. What's the average house price?")
print(f"      → ${housing_df['price'].mean():,.2f}")

print("\n   2. What's the price range?")
print(f"      → Min: ${housing_df['price'].min():,.2f}")
print(f"      → Max: ${housing_df['price'].max():,.2f}")

print("\n   3. Which feature correlates most with price?")
correlations = housing_df.corr(numeric_only=True)['price'].sort_values(ascending=False)
print(correlations)

print("\n   4. How many houses per neighborhood?")
print(housing_df['neighborhood'].value_counts())

print("\n   5. Average price by neighborhood:")
avg_by_neighborhood = housing_df.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
print(avg_by_neighborhood)

# Visualize
print("\n📊 Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Price distribution
axes[0, 0].hist(housing_df['price'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(housing_df['price'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Sqft vs Price
axes[0, 1].scatter(housing_df['sqft'], housing_df['price'], alpha=0.5)
axes[0, 1].set_title('Square Footage vs Price')
axes[0, 1].set_xlabel('Square Feet')
axes[0, 1].set_ylabel('Price ($)')

# Age vs Price
axes[1, 0].scatter(housing_df['age'], housing_df['price'], alpha=0.5, color='green')
axes[1, 0].set_title('House Age vs Price')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].set_ylabel('Price ($)')

# Price by neighborhood
housing_df.boxplot(column='price', by='neighborhood', ax=axes[1, 1])
axes[1, 1].set_title('Price by Neighborhood')
axes[1, 1].set_xlabel('Neighborhood')
axes[1, 1].set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig('interactive_housing_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Saved: interactive_housing_analysis.png")
plt.close()

# ============================================================================
# SECTION 2: Churn Data Exploration
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Exploring the Customer Churn Dataset")
print("="*80)

# Load the data
churn_df = pd.read_csv("customers.csv")

print("\n📊 Dataset Overview:")
print(f"   Rows: {churn_df.shape[0]}")
print(f"   Columns: {churn_df.shape[1]}")
print(f"   Features: {list(churn_df.columns)}")

print("\n📝 First 10 rows:")
print(churn_df.head(10))

print("\n📈 Statistical Summary:")
print(churn_df.describe())

print("\n❓ Missing Values:")
missing = churn_df.isnull().sum()
print(missing[missing > 0])

print("\n👥 Questions to explore:")
print("   1. How many customers churned?")
churn_counts = churn_df['churn'].value_counts()
print(churn_counts)
print(f"   Churn Rate: {churn_df['churn'].mean()*100:.2f}%")

print("\n   2. Average age of churned vs stayed customers:")
print(churn_df.groupby('churn')['age'].mean())

print("\n   3. Average income of churned vs stayed customers:")
print(churn_df.groupby('churn')['income'].mean())

print("\n   4. Average tenure of churned vs stayed customers:")
print(churn_df.groupby('churn')['tenure'].mean())

print("\n   5. Churn rate by contract type:")
churn_by_contract = churn_df.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
print(churn_by_contract)

print("\n   6. Churn rate by payment method:")
churn_by_payment = churn_df.groupby('payment_method')['churn'].mean().sort_values(ascending=False)
print(churn_by_payment)

# Visualize
print("\n📊 Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Churn distribution
churn_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Churn Distribution')
axes[0, 0].set_xlabel('Churn (0=Stay, 1=Leave)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(['Stay', 'Leave'], rotation=0)

# Age distribution by churn
churn_df.boxplot(column='age', by='churn', ax=axes[0, 1])
axes[0, 1].set_title('Age by Churn Status')
axes[0, 1].set_xlabel('Churn (0=Stay, 1=Leave)')
axes[0, 1].set_ylabel('Age')

# Income distribution by churn
churn_df.boxplot(column='income', by='churn', ax=axes[0, 2])
axes[0, 2].set_title('Income by Churn Status')
axes[0, 2].set_xlabel('Churn (0=Stay, 1=Leave)')
axes[0, 2].set_ylabel('Income ($)')

# Tenure distribution by churn
churn_df.boxplot(column='tenure', by='churn', ax=axes[1, 0])
axes[1, 0].set_title('Tenure by Churn Status')
axes[1, 0].set_xlabel('Churn (0=Stay, 1=Leave)')
axes[1, 0].set_ylabel('Tenure (months)')

# Churn by contract type
churn_by_contract.plot(kind='bar', ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Churn Rate by Contract Type')
axes[1, 1].set_xlabel('Contract Type')
axes[1, 1].set_ylabel('Churn Rate')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

# Churn by payment method
churn_by_payment.plot(kind='bar', ax=axes[1, 2], color='purple')
axes[1, 2].set_title('Churn Rate by Payment Method')
axes[1, 2].set_xlabel('Payment Method')
axes[1, 2].set_ylabel('Churn Rate')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('interactive_churn_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Saved: interactive_churn_analysis.png")
plt.close()

# ============================================================================
# SECTION 3: Key Insights
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Key Insights from Exploration")
print("="*80)

print("\n🏠 HOUSING INSIGHTS:")
print("   1. Square footage is the strongest predictor of price")
print(f"      Correlation: {housing_df['sqft'].corr(housing_df['price']):.3f}")

print("\n   2. Newer houses tend to be more expensive")
print(f"      Age-Price correlation: {housing_df['age'].corr(housing_df['price']):.3f}")

print("\n   3. Neighborhood matters!")
expensive_neighborhood = avg_by_neighborhood.index[0]
cheap_neighborhood = avg_by_neighborhood.index[-1]
print(f"      Most expensive: {expensive_neighborhood} (${avg_by_neighborhood.iloc[0]:,.0f})")
print(f"      Least expensive: {cheap_neighborhood} (${avg_by_neighborhood.iloc[-1]:,.0f})")

print("\n👥 CHURN INSIGHTS:")
print("   1. Month-to-month contracts have highest churn")
print(f"      Rate: {churn_by_contract.iloc[0]*100:.1f}%")

print("\n   2. Customers with longer tenure churn less")
avg_tenure_stay = churn_df[churn_df['churn']==0]['tenure'].mean()
avg_tenure_churn = churn_df[churn_df['churn']==1]['tenure'].mean()
print(f"      Staying customers: {avg_tenure_stay:.1f} months average")
print(f"      Churning customers: {avg_tenure_churn:.1f} months average")

print("\n   3. Electronic check users churn more")
print(f"      Rate: {churn_by_payment.iloc[0]*100:.1f}%")

print("\n💡 WHAT THIS MEANS FOR MODELING:")
print("   • Housing: Strong linear relationships → Linear models will work well")
print("   • Churn: Complex patterns → May need ensemble methods")
print("   • Both: Missing values need careful handling")
print("   • Churn: Class imbalance → Use F1-score, not just accuracy")

print("\n" + "="*80)
print("✅ Exploration Complete!")
print("="*80)
print("\n📁 Files Created:")
print("   • interactive_housing_analysis.png")
print("   • interactive_churn_analysis.png")
print("\n💡 Next Steps:")
print("   • Run supervised_ml_complete.py to train models")
print("   • Experiment with different algorithms")
print("   • Try feature engineering to improve performance")
print("="*80)
