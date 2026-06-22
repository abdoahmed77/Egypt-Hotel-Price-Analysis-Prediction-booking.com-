# 🏨 Egypt Hotel Price Analysis & Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600?style=for-the-badge&logo=xgboost&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-4.x-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**An end-to-end data science project — from scraping live hotel data to training machine learning regressors that predict nightly prices across Egypt.**

[📊 Dataset](#-dataset-information) • [🔬 EDA](#-exploratory-data-analysis) • [🤖 Models](#-machine-learning-models) • [📈 Results](#-model-evaluation) • [🚀 Usage](#-usage)

</div>

---

## 📌 Project Overview

This project builds a complete data science pipeline around the Egyptian hotel market. Live hotel listings were scraped from **Booking.com** across **17 Egyptian cities**, covering everything from five-star Hurghada resorts to budget hostels in Upper Egypt. The collected data was then cleaned, explored through rich visualisations, and used to train **seven regression models** that predict the nightly room price in Egyptian Pounds (EGP).

**Objectives:**

- Design and execute a production-grade Selenium web scraper capable of harvesting hotel attributes at scale from Booking.com.
- Conduct thorough exploratory data analysis to uncover pricing dynamics across cities, property types, amenities, and booking policies.
- Engineer meaningful features from raw scraped text (amenity flags, distance normalisation, meal-plan encoding).
- Train and compare a diverse set of regression algorithms, evaluate them rigorously, and identify the best performer.
- Persist the cleaned dataset to a **SQLite** database and demonstrate analytical SQL queries.

---

## 📂 Dataset Information

| Attribute | Value |
|---|---|
| 📋 **Total Records** | 1,040 hotel listings |
| 🗂️ **Total Features** | 19 raw columns (expanded to 303 after feature engineering) |
| 🌍 **Data Source** | [Booking.com](https://www.booking.com) |
| 🤖 **Collection Method** | Automated Selenium web scraper (`undetected-chromedriver`) |
| 📅 **Scraped Date** | May 9, 2026 (check-in: June 1 → June 2, 2026) |
| 💱 **Currency** | Egyptian Pound (EGP) |
| 🗺️ **Geographic Coverage** | 17 cities across Egypt |
| 🏙️ **Cities Covered** | Cairo, Alexandria, Hurghada, Sharm El Sheikh, Luxor, Aswan, El Gouna, Dahab, Marsa Alam, Siwa Oasis, Ain Sokhna, El Alamein, Marsa Matruh, Fayoum, Al Minya, Asyut, Sohag |
| 👥 **Booking Config** | 2 adults · 0 children · 1 room · 1 night |

### Price Statistics (Raw Dataset)

| Metric | Value |
|---|---|
| Minimum | 264 EGP |
| 25th Percentile | 1,845 EGP |
| Median | 3,427 EGP |
| Mean | 5,378 EGP |
| 75th Percentile | 6,140 EGP |
| Maximum | 474,480 EGP |

> ℹ️ The extremely high maximum reflects a small number of luxury outlier listings. Price outliers were removed during preprocessing (1st–80th percentile filter), capping the modelling dataset at **821 records** with a mean of **3,077 EGP** and a range of **580–6,992 EGP**.

---

## 🕷️ Web Scraping Pipeline

The scraper (`Web_Scrapping.py`) is built on **Selenium** with **undetected-chromedriver**, designed to bypass bot-detection mechanisms on Booking.com.

### Architecture

```
Build URL per city  ──▶  Launch undetected Chrome  ──▶  Wait for property cards
       │                                                          │
       ▼                                                          ▼
Inject anti-detection JS  ◀──────────────────────  Extract all hotel cards
(navigator.webdriver, plugins, languages)                        │
                                                                  ▼
                                              Per-card extraction loop:
                                              • Hotel name & URL
                                              • Price (regex-cleaned)
                                              • Guest rating & review count
                                              • Location & distance from centre
                                              • Star rating (counted from DOM)
                                              • Property type (inferred)
                                              • Room type, meal plan
                                              • Free cancellation flag
                                              • Amenities (scroll + lazy load)
                                                        │
                                                        ▼
                                               Save per-city CSV
                                                        │
                                                        ▼
                                          Merge ──▶ all_egypt_hotels.csv
```

### Key Technical Decisions

| Decision | Reason |
|---|---|
| `undetected-chromedriver` | Evades Cloudflare and Booking.com bot fingerprinting |
| Anti-detection JS injection | Masks `navigator.webdriver`, spoof plugins and language headers |
| `WebDriverWait` (60 s timeout) | Handles slow page renders without hard sleeps |
| `execute_script` scrolling | Triggers lazy-loaded amenity elements before extraction |
| Regex price cleaning (`re.sub(r'[^\d]', '', ...)`) | Strips currency symbols and formatting from price strings |
| Per-city CSV + merged master CSV | Allows partial recovery if a city scrape fails mid-run |
| Random sleep (`randint(5, 10)`) | Human-like pacing to reduce rate-limit risk |

---

## 📋 Dataset Features

### Raw Columns

| Column | Type | Description |
|---|---|---|
| `city` | `str` | Egyptian city where the hotel is located |
| `hotel_name` | `str` | Official listing name on Booking.com |
| `price` | `int64` | Nightly price in EGP (1 night, 2 adults) |
| `currency` | `str` | Always `EGP` for this dataset |
| `num_adults` | `int64` | Number of adults in the booking (2) |
| `num_children` | `int64` | Number of children in the booking (0) |
| `num_nights` | `int64` | Duration of stay in nights (1) |
| `rating` | `float64` | Guest rating score (0–10); 95 nulls |
| `reviews_count` | `float64` | Total number of guest reviews; 95 nulls |
| `location` | `str` | Sub-location or neighbourhood label |
| `distance_from_center` | `str` | Raw distance string (e.g., "1.5 km", "300 m"); 158 nulls |
| `star_rating` | `float64` | Official star classification (1–5); 292 nulls |
| `property_type` | `str` | Property category (Hotel, Resort, Apartment, etc.) |
| `room_type` | `str` | Recommended room name from listing card |
| `free_cancellation` | `str` | `"Yes"` / `"No"` cancellation policy |
| `meal_plan` | `str` | Included meals (e.g., Breakfast, All-inclusive); 511 nulls |
| `amenities` | `str` | Comma-separated amenity list; 378 nulls |
| `hotel_url` | `str` | Direct Booking.com listing URL |
| `scraped_date` | `str` | Date the record was scraped |

### Engineered Features (Post-Processing)

| Feature | Description |
|---|---|
| `distance_from_center` | Parsed to `float` in km (converted metres → km) |
| `free_cancellation` | Binary encoded: `Yes → 1`, `No → 0` |
| `amenities_missing` | Binary flag: 1 if amenities field was null |
| `has_wifi` | 1 if "Free Wifi" present in amenities |
| `has_spa` | 1 if "Spa" present in amenities |
| `has_parking` | 1 if "Free parking" present in amenities |
| `has_restaurant` | 1 if "Restaurant" present in amenities |
| `has_bar` | 1 if "Bar" present in amenities |
| `has_pool` | 1 if "Heated pool" present in amenities |
| `has_jacuzzi` | 1 if "Hot tub/Jacuzzi" present in amenities |
| OHE city features | One-hot encoding of `city` (17 categories) |
| OHE room_type features | One-hot encoding of `room_type` |
| OHE property_type features | One-hot encoding of `property_type` (8 categories) |
| OHE meal_plan features | One-hot encoding of `meal_plan` (5 categories incl. Room Only) |

---

## 🔬 Exploratory Data Analysis

The EDA section (Section 3 of the notebook) includes **7 major visualisation groups**, each accompanied by analytical markdown insights.

### 3.1 Price Distribution
Dual histograms compare the raw price distribution against its log-transformed form. The original distribution is heavily right-skewed (a small number of luxury listings inflate the tail), while `log1p(price)` produces a near-normal, symmetric distribution — confirming that **log transformation is the correct target encoding** for regression.

### 3.2 Median Price by City
A ranked bar chart reveals a clear geographic pricing hierarchy:

| Rank | City | Median Price |
|---|---|---|
| 1 | El Alamein | 7,503 EGP |
| 2 | Sharm El Sheikh | 5,536 EGP |
| 3 | Hurghada | 5,272 EGP |
| 4 | Marsa Alam | 5,061 EGP |
| 5 | El Gouna | 4,840 EGP |

> **Finding:** Red Sea and North Coast coastal resorts command a significant premium over Upper Egypt heritage cities (Luxor, Aswan, Asyut), reflecting the dominance of resort-style international tourism versus domestic budget travel inland.

### 3.3 Price by Property Type
A box-plot (outliers suppressed) paired with a pie chart shows the property mix:
- **Hotels** are the most common listing type (47.1%), followed by Apartments (24.3%) and Resorts (16.1%).
- **Resorts** and **Hotels** occupy the highest price ranges; **Hostels** and **Guest Houses** serve the budget end.
- **Villas** show the widest interquartile range, reflecting a mix of modest rentals and ultra-luxury private properties.

### 3.4 Rating & Star Rating Distributions
- Average guest score: **8.45 / 10** across 945 rated hotels; distribution is left-skewed with a dense cluster at 8–9.
- Star ratings concentrate at **3★ (269)** and **4★ (257)**, with 5★ properties at 209.

### 3.5 Price vs Star Rating
A step-wise bar chart of median price by star category confirms the expected monotonic increase — 5★ properties have a median price roughly **4× higher** than 2★ properties.

### 3.6 Amenity Adoption & Price Impact
Two side-by-side charts reveal both amenity prevalence and the price premium each amenity carries:
- **Spa** and **Jacuzzi** show the largest price premium (hotels with these amenities are significantly more expensive than those without).
- **Free Parking** is the most common amenity but carries the smallest price delta, confirming it's offered across all market tiers.

### 3.7 Meal Plan & Cancellation Policy Impact
- Full-board and all-inclusive properties are systematically more expensive, as these are typically resort-style offerings.
- Hotels offering **free cancellation** have slightly higher median prices, suggesting premium properties use flexible policies as a competitive advantage to attract bookings.

---

## 🧹 Data Preprocessing

All preprocessing steps are implemented in **Section 2** and **Section 4.1** of the notebook.

### Step-by-Step Cleaning Pipeline

**1. Drop Irrelevant Columns**
```python
df.drop(columns=['hotel_url', 'scraped_date', 'currency'], inplace=True)
```
These columns carry no predictive signal for price modelling.

**2. Parse Distance to Numeric (km)**
```python
def clean_distance(value):
    if 'km' in value: return float(value.split('km')[0].strip())
    elif 'm' in value: return float(value.split('m')[0].strip()) / 1000
    return np.nan
```
Converts raw strings like `"1.5 km"` or `"300 m"` to a uniform float in kilometres.

**3. Encode Free Cancellation**
```python
df['free_cancellation'] = df['free_cancellation'].map({'Yes': 1, 'No': 0})
```

**4. Impute Missing Values**
| Column | Strategy |
|---|---|
| `meal_plan` | Fill with `"Room Only"` (explicit category for missing) |
| `rating` | Fill with column median |
| `reviews_count` | Fill with column median |
| `star_rating` | Fill with column median |
| `distance_from_center` | Fill with column median |

**5. Amenity Feature Engineering**
Seven binary columns extracted from the raw comma-separated `amenities` string via `str.contains()`, plus an `amenities_missing` flag. The original `amenities` column is then dropped.

**6. Price Outlier Removal**
```python
low  = df['price'].quantile(0.01)
high = df['price'].quantile(0.80)
df   = df[(df['price'] >= low) & (df['price'] <= high)]
```
Clips extreme luxury outliers, reducing the modelling set from 1,040 to **821 records** and focusing predictions on the realistic market range (580–6,992 EGP).

**7. Log-Transform Target Variable**
```python
y_train = np.log1p(y_train)
y_test  = np.log1p(y_test)
```
Transforms the skewed price target to a near-normal distribution, improving linear model fit and error homoscedasticity. Predictions are back-transformed via `np.expm1()` before computing EGP-denominated metrics.

**8. Feature Encoding & Scaling**
- **Categorical columns** (`city`, `room_type`, `property_type`, `meal_plan`): `OneHotEncoder(handle_unknown='ignore')` → 286 OHE columns.
- **Numerical columns** (16 features): `StandardScaler` (zero mean, unit variance).
- Final feature matrix: **303 columns**.

**9. Drop Non-Predictive Text Columns**
```python
df_model = df.drop(columns=['hotel_name', 'location'])
```

**10. Train/Test Split**
80/20 split with `random_state=42` → 656 training samples, 165 test samples.

---

## 🤖 Machine Learning Models

Seven regression algorithms were trained on the log-transformed price target.

| # | Model | Key Hyperparameters |
|---|---|---|
| 1 | **Linear Regression** | Default (OLS) |
| 2 | **Ridge Regression** | `alpha=1.0` |
| 3 | **Lasso Regression** | `alpha=0.001` |
| 4 | **Decision Tree Regressor** | `max_depth=10`, `random_state=42` |
| 5 | **Random Forest Regressor** | `n_estimators=200`, `random_state=42` |
| 6 | **Gradient Boosting Regressor** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=5` |
| 7 | **XGBoost Regressor** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |

All models are trained on the same log-transformed target and evaluated by back-transforming predictions to EGP before computing MAE and RMSE, ensuring metrics are interpretable in real monetary terms.

---

## 📈 Model Evaluation

> ⚠️ **Note:** R² is measured on the **log-price** scale (model optimisation target). MAE and RMSE are measured on the **original EGP scale** (back-transformed via `np.expm1()`) for real-world interpretability.

| Model | R² (log scale) | MAE (EGP) | RMSE (EGP) |
|---|:---:|:---:|:---:|
| Linear Regression | 0.3793 | 1,091 | 1,489 |
| Ridge Regression | 0.4599 | 975 | 1,292 |
| **Lasso Regression** ⭐ | **0.4793** | **933** | **1,241** |
| Decision Tree | 0.2181 | 1,163 | 1,550 |
| Random Forest | 0.4194 | 990 | 1,297 |
| Gradient Boosting | 0.4424 | 948 | 1,258 |
| XGBoost | 0.4626 | 929 | 1,229 |

> The notebook also generates a horizontal bar chart comparing all models on both R² and MAE, with colour-coded performance tiers (green / orange / red), and a Predicted vs Actual scatter plot with residual distribution histogram for the best model.

---

## 🏆 Best Model

**Lasso Regression** (`alpha=0.001`) achieved the highest R² of **0.4793** on the log-price test set, with a Mean Absolute Error of **933 EGP** on actual nightly prices.

### Why Lasso Won

The high-dimensional feature space (303 columns after one-hot encoding) strongly favours regularised linear models. Lasso's L1 penalty performs automatic **feature selection** by driving the coefficients of irrelevant or redundant OHE dummies to exactly zero, preventing the overfitting that linear regression exhibits and outperforming the heavier tree-ensemble models that require substantially more data to generalise well in high-dimensional sparse spaces.

Notably, **XGBoost** achieved the lowest raw MAE (929 EGP) and RMSE (1,229 EGP) in EGP terms — making it the best model for absolute price error — but ranked second on R² (0.4626). Both models are strong candidates depending on whether ranking accuracy or absolute error minimisation is the priority.

### Residual Analysis

The residual distribution (plotted in the notebook) is approximately symmetric around zero, confirming the log-transform successfully addressed heteroscedasticity. Some right-skew in residuals reflects occasional luxury listings that still evade the outlier filter.

---

## 🎯 Feature Importance

Feature importances were extracted from both **Random Forest** and **XGBoost** (the two best tree-based models).

### Random Forest — Top 15 Features

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `reviews_count` | 0.1509 |
| 2 | `distance_from_center` | 0.1410 |
| 3 | `star_rating` | 0.0937 |
| 4 | `rating` | 0.0871 |
| 5 | `has_wifi` | 0.0789 |
| 6 | `meal_plan_All-inclusive` | 0.0299 |
| 7 | `property_type_Resort` | 0.0254 |
| 8 | `city_El Gouna` | 0.0231 |
| 9 | `city_El Alamein` | 0.0151 |
| 10 | `meal_plan_Room Only` | 0.0131 |
| 11 | `city_Luxor` | 0.0131 |
| 12 | `free_cancellation` | 0.0113 |
| 13 | `room_type_Deluxe Double Room` | 0.0109 |
| 14 | `room_type_One-Bedroom Apartment` | 0.0106 |
| 15 | `city_Cairo` | 0.0098 |

### XGBoost — Top 10 Features

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `has_wifi` | 0.1317 |
| 2 | `meal_plan_All-inclusive` | 0.0480 |
| 3 | `property_type_Resort` | 0.0348 |
| 4 | `property_type_Inn` | 0.0316 |
| 5 | `star_rating` | 0.0313 |
| 6 | `city_El Gouna` | 0.0279 |
| 7 | `meal_plan_Breakfast & dinner included` | 0.0253 |
| 8 | `city_El Alamein` | 0.0208 |
| 9 | `city_Cairo` | 0.0178 |
| 10 | `city_Sohag` | 0.0159 |

> **Interpretation:** Both models agree that **numerical continuous features** (reviews count, distance, star rating, guest rating) dominate in tree-based importance, while **WiFi availability**, **all-inclusive meal plans**, **resort classification**, and **coastal city membership** are the most impactful categorical signals.

---

## 🗄️ Database Integration

After modelling, the cleaned dataset is persisted to a **SQLite** database (`egypt_hotels.db`) and interrogated with 10 analytical SQL queries, including:

```sql
-- Top 10 most expensive hotels
SELECT city, hotel_name, price FROM hotels ORDER BY price DESC LIMIT 10;

-- Average price per city
SELECT city, AVG(price) AS avg_price FROM hotels GROUP BY city ORDER BY avg_price DESC;

-- Hotels with pool
SELECT COUNT(*) AS hotels_with_pool FROM hotels WHERE has_pool = 1;

-- City with highest average rating
SELECT city, AVG(rating) AS avg_rating FROM hotels GROUP BY city ORDER BY avg_rating DESC LIMIT 1;

-- Hotels within 1 km of city centre
SELECT city, hotel_name, distance_from_center FROM hotels WHERE distance_from_center < 1;
```

This demonstrates the ability to integrate a **data engineering layer** into a data science workflow and perform ad-hoc analysis without reloading the full Python environment.

---

## 💡 Key Insights

1. **Geography is the strongest price driver.** Coastal North Coast and Red Sea destinations (El Alamein, Sharm El Sheikh, Hurghada) command median nightly rates 2–3× higher than Upper Egypt cities (Asyut, Sohag, Al Minya), reflecting entirely different tourism markets.

2. **Lasso Regression outperforms tree ensembles on this dataset.** High-dimensional sparse OHE features (303 columns, 821 samples) favour L1-regularised linear models that can zero out irrelevant dummies — a dataset size where ensemble trees tend to over-fit.

3. **Reviews count and distance from centre are the top predictive signals.** Hotels with many reviews (established, reputable properties) and those closer to city centres are strong proxies for quality and demand — both are captured in the top-2 Random Forest features.

4. **Spa and Jacuzzi are the most price-differentiating amenities.** Hotels offering these premium wellness facilities have significantly higher median prices than those without, while free parking carries virtually no price premium.

5. **56.7% of Egyptian hotels on Booking.com offer free cancellation**, and these properties trend slightly more expensive — likely because premium hotels use flexibility as a demand signal.

6. **Three-star and four-star properties dominate the Egyptian market** (269 and 257 listings respectively), with 5★ making up 27.7% — reflecting the maturity of Egypt's mid-market hotel sector.

7. **All-inclusive meal plans carry a large price premium** and are concentrated in Red Sea resort destinations, while the majority of city hotels (especially Upper Egypt) are sold as room-only.

8. **Average guest satisfaction is high across Egypt** — a mean rating of 8.45/10 across 945 reviewed hotels suggests that guests generally receive value for money, with over 65% of rated hotels scoring above 8.

9. **Villas exhibit the widest price variance**, mixing budget private rentals with ultra-luxury villas, making the property type categorisation particularly challenging for the models.

10. **XGBoost identifies WiFi as its single most important feature (0.13 importance)**, suggesting that in the Egyptian budget-to-mid market, WiFi availability is a strong proxy for modernisation and overall property quality.

---

## 🛠️ Technologies Used

### Core Libraries

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.x | Data loading, manipulation, and analysis |
| `numpy` | 1.x | Numerical operations and log transforms |
| `matplotlib` | 3.x | Base visualisation and custom formatting |
| `seaborn` | 0.13.x | Statistical plots (box plots, bar charts, histograms) |
| `scikit-learn` | 1.4+ | ML pipeline: preprocessing, models, metrics |
| `xgboost` | 2.x | XGBoost gradient boosting regressor |
| `sqlite3` | stdlib | Database creation and SQL querying |

### Scraping Stack

| Tool | Purpose |
|---|---|
| `selenium` | Browser automation and DOM interaction |
| `undetected-chromedriver` | Anti-bot Chrome driver |
| `re` | Regex-based price and rating extraction |
| `csv` | Per-city file output |

### Scikit-learn Components

| Component | Usage |
|---|---|
| `train_test_split` | 80/20 stratified split |
| `StandardScaler` | Normalise numerical features |
| `OneHotEncoder` | Encode categorical features |
| `LinearRegression` | OLS baseline |
| `Ridge` | L2 regularised linear |
| `Lasso` | L1 regularised linear (best model) |
| `DecisionTreeRegressor` | Single tree baseline |
| `RandomForestRegressor` | Bagging ensemble |
| `GradientBoostingRegressor` | Boosting ensemble |
| `r2_score`, `mean_absolute_error`, `mean_squared_error` | Evaluation metrics |

---

## 📁 Repository Structure

```
egypt-hotel-price-prediction/
│
├── 📓 Hotel_Price_Analysis_and_prediction.ipynb   # Main analysis notebook
├── 🕷️  Web_Scrapping.py                           # Selenium scraper
├── 📊 all_egypt_hotels.csv                        # Master scraped dataset (1,040 rows)
│
├── data/                                          # (optional) per-city CSVs
│   ├── cairo_hotels.csv
│   ├── hurghada_hotels.csv
│   ├── sharm_hotels.csv
│   └── ... (17 city files)
│
├── egypt_hotels.db                               # SQLite database (generated by notebook)
│
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/egypt-hotel-price-prediction.git
cd egypt-hotel-price-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.4
xgboost>=2.0
selenium>=4.0
undetected-chromedriver>=3.5
jupyter>=1.0
ipykernel>=6.0
```

### 4. (Scraper Only) Install ChromeDriver

The scraper uses `undetected-chromedriver`, which auto-manages ChromeDriver. Ensure **Google Chrome** is installed and its version matches the `version_main=147` argument in `setup_driver()`, or update that parameter to your installed Chrome version.

---

## 🚀 Usage

### Run the Web Scraper

```bash
python Web_Scrapping.py
```

- Iterates through all 17 configured Egyptian cities.
- Saves individual city CSVs and a combined `all_egypt_hotels.csv`.
- To scrape a subset of cities, edit the `CITIES` list at the bottom of the script.
- To change the booking date, update `CHECKIN` and `CHECKOUT` constants at the top.

> ⚠️ **Rate Limiting:** The scraper includes random delays (5–10 s per city) to avoid IP bans. Do not reduce these delays aggressively.

### Run the Analysis Notebook

```bash
jupyter notebook "Hotel_Price_Analysis_and_prediction.ipynb"
```

Then execute all cells sequentially (Kernel → Restart & Run All). The notebook will:
1. Load `all_egypt_hotels.csv`
2. Perform full preprocessing
3. Generate all EDA visualisations
4. Train and evaluate all 7 models
5. Display feature importances
6. Create the SQLite database and run SQL queries

---

## 🔮 Future Improvements

| Enhancement | Description |
|---|---|
| 📅 **Temporal Price Tracking** | Re-scrape weekly to build a time-series dataset and capture seasonal pricing dynamics (peak summer, Eid, Christmas). |
| 🔍 **Hyperparameter Tuning** | Apply `GridSearchCV` or `Optuna` to systematically optimise tree-ensemble hyperparameters (depth, learning rate, subsample). |
| 🧮 **Price-per-Star Normalisation** | Engineer a quality-adjusted price feature to better model the luxury vs. budget dimension. |
| 🗺️ **Geospatial Features** | Integrate Google Maps API distances to beaches, airports, or tourist attractions as continuous proximity features. |
| 🌐 **Multilingual Reviews** | Scrape and perform sentiment analysis on Arabic and English guest reviews to add NLP-derived reputation scores. |
| 🧩 **Stacking Ensemble** | Build a meta-learner (e.g., Ridge on top of Random Forest + XGBoost) to blend predictions from the best individual models. |
| 📱 **Streamlit Dashboard** | Deploy an interactive price prediction app where users input city, star rating, and amenities to receive an estimated nightly rate. |
| 📊 **Booking Rate Estimation** | Extend from price prediction to demand prediction by scraping availability data over time. |
| 🔧 **Pipeline Serialisation** | Wrap the full preprocessing + model pipeline in `sklearn.Pipeline` and serialise with `joblib` for production deployment. |

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙌 Acknowledgements

- [**Booking.com**](https://www.booking.com) — data source for all hotel listings.
- [**Scikit-learn**](https://scikit-learn.org) — comprehensive ML ecosystem.
- [**XGBoost**](https://xgboost.readthedocs.io) — high-performance gradient boosting.
- [**undetected-chromedriver**](https://github.com/ultrafunkamsterdam/undetected-chromedriver) — anti-bot browser automation.

---

## 👥 Team

This project was built collaboratively by a team of four:

| Name | GitHub |
|---|---|
| **Abdalrhman Ahmed** | [@abdalrhman-ahmed](https://github.com/abdalrhman-ahmed) |
| **Mohammed Hazem** | [@mohammed-hazem](https://github.com/mohammed-hazem) |
| **Ali Sayed** | [@ali-sayed](https://github.com/ali-sayed) |
| **Ahmed Adel** | [@ahmed-adel](https://github.com/ahmed-adel) |

> 📝 **Note:** Replace each `@placeholder` above with the actual GitHub username for each team member.

---

<div align="center">

⭐ **If you found this project useful, please star the repository!** ⭐

Made with ❤️ and lots of ☕ by a team of 4 — Data scraped from Egypt's hotel market, May 2026

</div>
