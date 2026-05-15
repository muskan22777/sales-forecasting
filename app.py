"""
Sales Forecasting Web App
Author: Muskan
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold;
        color: #8B1A1A; text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-header {
        text-align: center; color: #666;
        font-size: 1rem; margin-bottom: 20px;
    }
    .metric-card {
        background: #f9f9f9; border-left: 4px solid #8B1A1A;
        padding: 15px; border-radius: 6px; margin: 5px 0;
    }
    .metric-label { font-size: 0.85rem; color: #666; }
    .metric-value { font-size: 1.6rem; font-weight: bold; color: #8B1A1A; }
    .section-title {
        font-size: 1.2rem; font-weight: bold;
        color: #8B1A1A; border-bottom: 2px solid #8B1A1A;
        padding-bottom: 4px; margin: 15px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Resources ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, encoders, features

df = load_data()
model, encoders, features = load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Sales Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict future sales using Machine Learning</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=80)
st.sidebar.title("🔧 Controls")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Sales", "📁 Raw Data"])

# ══════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # KPI Cards
    total_sales  = df['Total_Sales'].sum()
    total_units  = df['Units_Sold'].sum()
    avg_daily    = df.groupby('Date')['Total_Sales'].sum().mean()
    top_category = df.groupby('Product_Category')['Total_Sales'].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">💰 Total Revenue</div>
            <div class="metric-value">₹{total_sales/1e6:.1f}M</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📦 Total Units Sold</div>
            <div class="metric-value">{total_units:,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📅 Avg Daily Sales</div>
            <div class="metric-value">₹{avg_daily:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">🏆 Top Category</div>
            <div class="metric-value">{top_category}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Filters
    st.markdown('<div class="section-title">📅 Filter Data</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        year_sel = st.multiselect("Year", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    with col2:
        cat_sel = st.multiselect("Category", df['Product_Category'].unique(), default=list(df['Product_Category'].unique()))
    with col3:
        reg_sel = st.multiselect("Region", df['Region'].unique(), default=list(df['Region'].unique()))

    filtered = df[df['Year'].isin(year_sel) & df['Product_Category'].isin(cat_sel) & df['Region'].isin(reg_sel)]

    # Chart 1 – Monthly Trend
    st.markdown('<div class="section-title">📈 Monthly Sales Trend</div>', unsafe_allow_html=True)
    monthly = filtered.groupby(filtered['Date'].dt.to_period('M'))['Total_Sales'].sum()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values / 1000, color='#8B1A1A', linewidth=2, marker='o', markersize=3)
    ax.fill_between(monthly.index, monthly.values / 1000, alpha=0.15, color='#8B1A1A')
    ax.set_ylabel('Sales (₹ Thousands)')
    ax.set_title('Monthly Sales Trend', fontweight='bold')
    sns.despine()
    st.pyplot(fig)
    plt.close()

    # Chart 2 & 3
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🛍️ Sales by Category</div>', unsafe_allow_html=True)
        cat_data = filtered.groupby('Product_Category')['Total_Sales'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(cat_data.index, cat_data.values / 1000, color='#8B1A1A')
        ax.set_xlabel('Sales (₹ Thousands)')
        for bar, val in zip(bars, cat_data.values / 1000):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'₹{val:.0f}K', va='center', fontsize=8)
        sns.despine()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">🗺️ Sales by Region</div>', unsafe_allow_html=True)
        reg_data = filtered.groupby('Region')['Total_Sales'].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_list = ['#8B1A1A', '#C0392B', '#E74C3C', '#F1948A']
        wedges, texts, autotexts = ax.pie(
            reg_data.values, labels=reg_data.index,
            autopct='%1.1f%%', colors=colors_list,
            startangle=90, pctdistance=0.75)
        ax.set_title('Regional Sales Share', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Chart 4 – Day of week
    st.markdown('<div class="section-title">📆 Sales by Day of Week</div>', unsafe_allow_html=True)
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_data = filtered.groupby('Day_of_Week')['Total_Sales'].mean().reindex(day_order)
    fig, ax = plt.subplots(figsize=(10, 3))
    colors_bar = ['#E74C3C' if d in ['Saturday','Sunday'] else '#8B1A1A' for d in day_order]
    ax.bar(day_data.index, day_data.values, color=colors_bar)
    ax.set_ylabel('Avg Daily Sales (₹)')
    ax.set_title('Average Sales by Day of Week  (Red = Weekend)', fontweight='bold')
    sns.despine()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 2 – PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Sales":
    st.markdown('<div class="section-title">🔮 Predict Future Sales</div>', unsafe_allow_html=True)
    st.info("Fill in the details below to get a sales prediction from our ML model.")

    col1, col2 = st.columns(2)
    with col1:
        pred_date    = st.date_input("📅 Select Date", datetime.today())
        product_cat  = st.selectbox("🛍️ Product Category",
                                    ['Electronics','Clothing','Groceries','Furniture','Sports'])
        region       = st.selectbox("🗺️ Region", ['North','South','East','West'])
    with col2:
        unit_price   = st.number_input("💲 Unit Price (₹)", min_value=10, max_value=5000, value=500)
        units_sold   = st.number_input("📦 Units Sold", min_value=1, max_value=500, value=50)

    if st.button("🚀 Predict Sales", use_container_width=True):
        d = pd.Timestamp(pred_date)
        row = {
            'Product_Enc':   encoders['product'].transform([product_cat])[0],
            'Region_Enc':    encoders['region'].transform([region])[0],
            'Day':           d.day,
            'Month':         d.month,
            'Year':          d.year,
            'Day_of_Week_N': d.dayofweek,
            'Week_of_Year':  d.isocalendar()[1],
            'Is_Weekend':    int(d.dayofweek >= 5),
            'Is_Festival':   int(d.month in [10, 11, 12]),
            'Unit_Price':    unit_price,
            'Units_Sold':    units_sold,
        }
        X_pred  = pd.DataFrame([row])[features]
        prediction = model.predict(X_pred)[0]

        st.success(f"### 🎯 Predicted Sales: ₹{prediction:,.2f}")

        # Context
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekend Boost", "Yes ✅" if row['Is_Weekend'] else "No ❌")
        with col2:
            st.metric("Festival Season", "Yes 🎉" if row['Is_Festival'] else "No")
        with col3:
            avg = df[(df['Product_Category'] == product_cat) &
                     (df['Region'] == region)]['Total_Sales'].mean()
            diff = ((prediction - avg) / avg) * 100
            st.metric("vs Historical Avg", f"{diff:+.1f}%")

        # 7-day forecast
        st.markdown('<div class="section-title">📅 7-Day Forecast</div>', unsafe_allow_html=True)
        forecast = []
        for i in range(7):
            fd = d + timedelta(days=i)
            fr = {
                'Product_Enc':   row['Product_Enc'],
                'Region_Enc':    row['Region_Enc'],
                'Day':           fd.day,
                'Month':         fd.month,
                'Year':          fd.year,
                'Day_of_Week_N': fd.dayofweek,
                'Week_of_Year':  fd.isocalendar()[1],
                'Is_Weekend':    int(fd.dayofweek >= 5),
                'Is_Festival':   int(fd.month in [10, 11, 12]),
                'Unit_Price':    unit_price,
                'Units_Sold':    units_sold,
            }
            fp = model.predict(pd.DataFrame([fr])[features])[0]
            forecast.append({'Date': fd.strftime('%a, %d %b'), 'Predicted Sales (₹)': round(fp, 2)})

        fc_df = pd.DataFrame(forecast)
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = ['#E74C3C' if 'Sat' in r or 'Sun' in r else '#8B1A1A'
                      for r in fc_df['Date']]
        ax.bar(fc_df['Date'], fc_df['Predicted Sales (₹)'], color=bar_colors)
        ax.set_ylabel('Predicted Sales (₹)')
        ax.set_title('7-Day Sales Forecast', fontweight='bold')
        sns.despine()
        st.pyplot(fig)
        plt.close()
        st.dataframe(fc_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 – RAW DATA
# ══════════════════════════════════════════════════════════════
elif page == "📁 Raw Data":
    st.markdown('<div class="section-title">📁 Raw Dataset</div>', unsafe_allow_html=True)
    st.write(f"**Total Records:** {len(df):,}")
    col1, col2 = st.columns(2)
    with col1:
        cat_f = st.multiselect("Filter by Category", df['Product_Category'].unique())
    with col2:
        reg_f = st.multiselect("Filter by Region", df['Region'].unique())

    show = df.copy()
    if cat_f: show = show[show['Product_Category'].isin(cat_f)]
    if reg_f: show = show[show['Region'].isin(reg_f)]

    st.dataframe(show.head(500), use_container_width=True)

    csv = show.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sales_data_filtered.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("<center>Built with ❤️ by Muskan | Sales Forecasting ML Project</center>",
            unsafe_allow_html=True)
