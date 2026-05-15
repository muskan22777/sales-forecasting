# 📈 Sales Forecasting System using Machine Learning

> Predict future product sales using historical data and ML models (Linear Regression, Random Forest, Gradient Boosting) with an interactive Streamlit web app.

---

## 🖥️ Demo

| Dashboard | Prediction | Data Explorer |
|-----------|-----------|---------------|
| KPIs, trends, charts | Predict any date/product | Filter & download data |

---

## 🚀 Features

- ✅ **3 ML Models** — Linear Regression, Random Forest, Gradient Boosting
- ✅ **Interactive Dashboard** — KPIs, monthly trends, category & region breakdowns
- ✅ **Sales Predictor** — Enter date + product → get instant prediction
- ✅ **7-Day Forecast** — Visualise upcoming week's predicted sales
- ✅ **Data Explorer** — Filter, view, and download the dataset
- ✅ **Feature Engineering** — Weekend flag, festival season, week of year, etc.

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3.10+ |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```
sales_forecasting/
│
├── data/
│   └── sales_data.csv          # 21,900 rows of historical sales data
│
├── models/
│   ├── best_model.pkl           # Trained ML model
│   ├── label_encoders.pkl       # Category encoders
│   └── feature_names.pkl        # Feature list
│
├── notebooks/
│   └── eda.py                   # Exploratory Data Analysis
│
├── screenshots/
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── monthly_trend.png
│   ├── sales_by_category.png
│   └── feature_importance.png
│
├── app.py                       # Streamlit web application
├── train_model.py               # Model training script
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/muskan22777/sales-forecasting.git
cd sales-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ~35 | ~45 | ~0.85 |
| Random Forest | ~12 | ~18 | ~0.97 |
| **Gradient Boosting** | **~14** | **~20** | **~0.96** |

> ✅ **Random Forest** selected as the best model based on R² score.

---

## 🔍 Key Insights

- 📅 **Weekend sales** are ~18% higher than weekdays
- 🎉 **Oct–Dec (Festival season)** sees a 35% spike in sales
- 🛒 **Groceries** have the highest volume; **Electronics** has the highest revenue
- 📈 **Overall sales trend** is upward (~0.05 units/day growth)

---

## 👩‍💻 Author

**Muskan**
- 🎓 B.Tech Big Data Analytics — Chandigarh University (2023–2027)
- 🔗 [LinkedIn](https://linkedin.com/in/muskan-608a4a353/)
- 💻 [GitHub](https://github.com/muskan22777)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
