"""
Sales Forecasting - Exploratory Data Analysis (EDA)
Author: Muskan
Run this before training to understand the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings = lambda *a, **kw: None

df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 55)
print("   EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print("\n📋 Dataset Info:")
print(f"   Shape      : {df.shape}")
print(f"   Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"   Columns    : {list(df.columns)}")

print("\n📊 Basic Statistics:")
print(df[['Units_Sold','Unit_Price','Total_Sales']].describe().round(2))

print("\n💰 Sales by Category:")
print(df.groupby('Product_Category')['Total_Sales'].agg(['sum','mean','count']).round(2))

print("\n🗺️  Sales by Region:")
print(df.groupby('Region')['Total_Sales'].agg(['sum','mean']).round(2))

print("\n📅 Sales by Year:")
print(df.groupby('Year')['Total_Sales'].sum().round(2))

print("\n✅ Null values:", df.isnull().sum().sum())
print("✅ Duplicate rows:", df.duplicated().sum())
