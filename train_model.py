"""
Sales Forecasting - Model Training
Author: Muskan
Description: Train ML models (Linear Regression, Random Forest, XGBoost)
             to forecast future sales based on historical data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SALES FORECASTING - MODEL TRAINING")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"\n✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ── 2. Feature Engineering ────────────────────────────────────
print("\n🔧 Engineering features...")
df['Day']          = df['Date'].dt.day
df['Month']        = df['Date'].dt.month
df['Year']         = df['Date'].dt.year
df['Day_of_Week_N'] = df['Date'].dt.dayofweek
df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
df['Is_Weekend']   = (df['Day_of_Week_N'] >= 5).astype(int)
df['Is_Festival']  = df['Month'].isin([10, 11, 12]).astype(int)

# Encode categoricals
le_product = LabelEncoder()
le_region  = LabelEncoder()
df['Product_Enc'] = le_product.fit_transform(df['Product_Category'])
df['Region_Enc']  = le_region.fit_transform(df['Region'])

# Save encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump({'product': le_product, 'region': le_region}, f)

# ── 3. Prepare X, y ───────────────────────────────────────────
features = ['Product_Enc', 'Region_Enc', 'Day', 'Month', 'Year',
            'Day_of_Week_N', 'Week_of_Year', 'Is_Weekend',
            'Is_Festival', 'Unit_Price', 'Units_Sold']

X = df[features]
y = df['Total_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4. Train Models ───────────────────────────────────────────
print("\n🤖 Training models...")

models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds,
                     'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"   {name:25s} → MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | R²: {r2:.4f}")

# ── 5. Save Best Model ────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)
print(f"\n💾 Best model saved: {best_name} (R² = {results[best_name]['R2']:.4f})")

# ── 6. Plots ──────────────────────────────────────────────────
print("\n📊 Generating plots...")
sns.set_style("whitegrid")
palette = '#8B1A1A'

# Plot 1 – Model Comparison
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
r2s   = [results[n]['R2'] for n in names]
bars  = ax.barh(names, r2s, color=['#8B1A1A', '#C0392B', '#E74C3C'])
ax.set_xlabel('R² Score')
ax.set_title('Model Comparison – R² Score', fontweight='bold')
ax.set_xlim(0, 1.05)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('screenshots/model_comparison.png', dpi=150)
plt.close()

# Plot 2 – Actual vs Predicted
best_preds = results[best_name]['preds']
fig, ax = plt.subplots(figsize=(8, 5))
sample = np.random.choice(len(y_test), 200, replace=False)
ax.scatter(y_test.iloc[sample], best_preds[sample],
           alpha=0.5, color='#8B1A1A', edgecolors='white', linewidth=0.3)
mn = min(y_test.min(), best_preds.min())
mx = max(y_test.max(), best_preds.max())
ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label='Perfect Prediction')
ax.set_xlabel('Actual Sales (₹)')
ax.set_ylabel('Predicted Sales (₹)')
ax.set_title(f'Actual vs Predicted – {best_name}', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('screenshots/actual_vs_predicted.png', dpi=150)
plt.close()

# Plot 3 – Monthly Sales Trend
monthly = df.groupby(['Year', 'Month'])['Total_Sales'].sum().reset_index()
monthly['Period'] = pd.to_datetime(
    monthly[['Year', 'Month']].assign(day=1))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly['Period'], monthly['Total_Sales']/1000,
        color='#8B1A1A', linewidth=2, marker='o', markersize=3)
ax.fill_between(monthly['Period'], monthly['Total_Sales']/1000,
                alpha=0.15, color='#8B1A1A')
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales (₹ Thousands)')
ax.set_title('Monthly Sales Trend (2022–2024)', fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/monthly_trend.png', dpi=150)
plt.close()

# Plot 4 – Sales by Category
cat_sales = df.groupby('Product_Category')['Total_Sales'].sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
cat_sales.plot(kind='barh', ax=ax,
               color=['#E74C3C','#C0392B','#A93226','#8B1A1A','#641010'])
ax.set_xlabel('Total Sales (₹)')
ax.set_title('Total Sales by Product Category', fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/sales_by_category.png', dpi=150)
plt.close()

# Plot 5 – Feature Importance (Random Forest)
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind='barh', ax=ax, color='#8B1A1A')
ax.set_title('Feature Importance – Random Forest', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=150)
plt.close()

print("   ✅ All plots saved to screenshots/")
print("\n" + "="*60)
print("   TRAINING COMPLETE!")
print("="*60)
print(f"\n   Best Model : {best_name}")
print(f"   R² Score   : {results[best_name]['R2']:.4f}")
print(f"   MAE        : ₹{results[best_name]['MAE']:.2f}")
print(f"   RMSE       : ₹{results[best_name]['RMSE']:.2f}")
