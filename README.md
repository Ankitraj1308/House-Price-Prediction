# 🏡 House Price Prediction Using Machine Learning

<p align="center">
  <img src="static/images/house_3d_render.jpg" alt="House Price Prediction 3D Isometric AI Model" width="550" style="border-radius: 24px; box-shadow: 0 25px 50px -12px rgba(15, 39, 68, 0.25);" />
</p>

An end-to-end, production-grade Machine Learning web application designed to predict residential real-estate market values. Built with **Scikit-Learn**, **Flask**, and **Bootstrap 5**, this application demonstrates a complete machine learning lifecycle: data preprocessing, exploratory data analysis (EDA), feature engineering, 4-model benchmarking, leakage-free pipeline design, and interactive web deployment.

![Model Test R2 Score](https://img.shields.io/badge/Test%20R%C2%B2%20Score-91.59%25-16a34a?style=for-the-badge&logo=scikit-learn)
![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask)
![Dataset](https://img.shields.io/badge/Dataset-Ames%20Housing%20(1%2C458%20records)-2563eb?style=for-the-badge)

---

## 📑 Table of Contents
- [1. Project Overview & Problem Statement](#1-project-overview--problem-statement)
- [2. System Architecture](#2-system-architecture)
- [3. Key Objectives](#3-key-objectives)
- [4. Technologies Used](#4-technologies-used)
- [5. Dataset & Feature Descriptions](#5-dataset--feature-descriptions)
- [6. Machine Learning Workflow & Pipeline](#6-machine-learning-workflow--pipeline)
- [7. Empirical Model Evaluation & Comparison](#7-empirical-model-evaluation--comparison)
- [8. Web Application Features](#8-web-application-features)
- [9. Project Structure & File Guide](#9-project-structure--file-guide)
- [10. Step-by-Step Installation & Run Guide (macOS)](#10-step-by-step-installation--run-guide-macos)
- [11. REST API Documentation](#11-rest-api-documentation)
- [12. Beginner's Guide: How the Model Makes Predictions](#12-beginners-guide-how-the-model-makes-predictions)
- [13. Future Improvements](#13-future-improvements)

---

## 1. Project Overview & Problem Statement

### The Problem
Traditional residential property appraisals are manual, time-consuming, and subject to individual cognitive biases. Home buyers, sellers, and mortgage lenders often lack transparent, data-driven valuation benchmarks that account for the non-linear interplay between physical size, construction quality, age, and neighborhood location.

### The Solution
This project automates property appraisal by training multiple regression models on authentic historical sales data. It automatically evaluates candidate algorithms, identifies the most generalizable model (**Gradient Boosting Regressor**), saves the end-to-end preprocessing pipeline, and wraps it in a responsive, real-estate-themed Flask web application that delivers instant property valuations in both **Indian Rupees (₹)** and **US Dollars ($)**.

---

## 2. System Architecture

```mermaid
flowchart TD
    A[Ames Housing Dataset: 1,460 Records] --> B[Data Cleansing & Outlier Filtering]
    B --> C[80:20 Train-Test Split]
    C --> D[Leakage-Free Preprocessing Pipeline\n- Median Imputer + StandardScaler\n- Most-Frequent Imputer + OneHotEncoder]
    D --> E[Benchmark 4 Algorithms:\n1. Linear Regression\n2. Decision Tree Regressor\n3. Random Forest Regressor\n4. Gradient Boosting Regressor]
    E --> F[Automated Best Model Selection\nGradient Boosting: Test R² = 91.59%]
    F --> G[Serialized Pipeline: models/best_model.joblib]
    F --> H[Metrics Metadata: models/model_metrics.json]
    B --> I[EDA Figures: visualizations/*.png]
    G --> J[Flask Web Server: app.py]
    H --> J
    J --> K[Interactive Web Pages & REST API:\n- Home / Introduction\n- Valuation Form with Presets\n- Result Card in INR (₹) & USD ($)\n- Interactive Chart.js Dashboard]
```

---

## 3. Key Objectives

1. **Clean Data Preprocessing:** Clean real-world housing data, handle missing values, and filter extreme non-arm’s length outliers without data leakage.
2. **Exploratory Data Analysis:** Generate 6 high-resolution statistical visualizations highlighting distribution skewness, correlations, and feature importances.
3. **Rigorous Machine Learning Benchmark:** Train, tune, and compare four regression models using $R^2$, MAE, MSE, and RMSE metrics on an unseen test set.
4. **Interactive Web Deployment:** Build a production-grade Flask application with a modern real-estate theme, live presets, client-side/server-side validation, and currency conversion.
5. **Portfolio-Quality Codebase:** Well-structured, fully documented, modular Python code ready for code review and demonstrations.

---

## 4. Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.13 | Primary development language |
| **Data Processing** | Pandas 3.1, NumPy 2.5 | Vectorized computation, dataframes, cleaning |
| **Machine Learning** | Scikit-Learn 1.9 | Pipelines, `ColumnTransformer`, regression models, metrics |
| **Serialization** | Joblib 1.6 | Serializing the fitted end-to-end pipeline |
| **Visualization** | Matplotlib 3.11, Seaborn 0.13 | High-resolution publication-quality EDA plots |
| **Backend** | Flask 3.1, Werkzeug | Lightweight WSGI web framework and routing |
| **Frontend** | HTML5, CSS3, JavaScript (ES6) | Responsive UI, interactive form autofill, validation |
| **UI Framework** | Bootstrap 5.3, Bootstrap Icons | Grid system, modern components, clean aesthetics |
| **Interactive Charts** | Chart.js 4.4 | Dynamic dual-axis benchmark and feature importance charts |

---

## 5. Dataset & Feature Descriptions

The model is trained on the renowned **Ames Housing Dataset** compiled by Professor Dean De Cock (Truman State University). The dataset records detailed physical, architectural, and geographic characteristics of 1,458 residential properties in Ames, Iowa.

### 11 Evaluated Features

| Feature Name | Type | Range / Options | Description |
| :--- | :--- | :--- | :--- |
| `OverallQual` | Integer | 1 to 10 | Rates the overall material and finish quality of the house |
| `GrLivArea` | Continuous | 300 – 10,000 sq ft | Above-grade (ground) living area in square feet |
| `BedroomAbvGr` | Discrete | 1 to 6+ | Number of bedrooms situated above ground level |
| `FullBath` | Discrete | 1 to 4+ | Number of full bathrooms above ground level |
| `Floors` | Numeric | 1.0, 1.5, 2.0, 2.5 | Number of stories/levels (derived from `HouseStyle`) |
| `GarageCars` | Discrete | 0 to 4 cars | Size of the garage in vehicle capacity |
| `YearBuilt` | Integer | 1870 – 2026 | Original construction year |
| `TotalBsmtSF` | Continuous | 0 – 8,000 sq ft | Total square footage of basement area |
| `LotArea` | Continuous | 500 – 150,000 sq ft | Lot / land plot size in square feet |
| `OverallCond` | Integer | 1 to 10 | Rates the current upkeep and physical maintenance condition |
| `Neighborhood` | Categorical | 25 Ames locations | Physical location within city limits (e.g., CollgCr, NoRidge, OldTown) |

**Target Variable:** `SalePrice` — Actual selling price in US Dollars ($).

---

## 6. Machine Learning Workflow & Pipeline

### 1. Data Cleaning & Outlier Removal
Following author Dean De Cock's documented recommendation, 2 extreme atypical outliers (`GrLivArea > 4000` sq ft with `SalePrice < $300,000`) representing non-arm's length agricultural/partial sales were removed, preserving genuine residential patterns.

### 2. Leakage-Free Preprocessing Pipeline
To prevent data snooping and data leakage:
- The dataset is split **80% Training (1,166 samples)** and **20% Testing (292 samples)** with a fixed random seed (`random_state=42`).
- A `ColumnTransformer` is used:
  - **Numeric Features Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - **Categorical Features Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Preprocessors are fitted **only** on the training partition and transformed onto the testing partition.

---

## 7. Empirical Model Evaluation & Comparison

All four algorithms were trained using identical preprocessing splits and evaluated on unseen test data:

| Model Name | Test $R^2$ Score | Train $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **91.59%** | **97.50%** | **\$15,020** | **\$21,549** | ★ **Selected Winner** |
| Random Forest Regressor | 88.28% | 98.21% | \$17,619 | \$25,443 | Evaluated |
| Linear Regression | 86.08% | 86.60% | \$20,225 | \$27,729 | Evaluated |
| Decision Tree Regressor | 75.17% | 100.00% | \$27,240 | \$37,032 | Overfits (Train $R^2 = 1.0$) |

### Metric Definitions
- **$R^2$ Score (Coefficient of Determination):** Measures the proportion of variance in house prices explained by the features. A score of **91.59%** means the model explains over 91% of price variability on unseen properties.
- **Mean Absolute Error (MAE):** The average absolute dollar error. On average, predictions deviate by only **\$15,020** on houses averaging \$180,000+.
- **Root Mean Squared Error (RMSE):** Measures standard deviation of residuals, penalizing large mistakes. The winning RMSE is **\$21,549**.

### Top Predictive Features
1. **Overall Quality (`OverallQual`):** 55.24% relative importance
2. **Above Ground Living Area (`GrLivArea`):** 17.80% relative importance
3. **Total Basement Area (`TotalBsmtSF`):** 11.55% relative importance
4. **Garage Capacity (`GarageCars`):** 4.48% relative importance
5. **Year Built (`YearBuilt`):** 4.42% relative importance
6. **Lot Area (`LotArea`):** 2.93% relative importance

---

## 8. Web Application Features

### 1. Landing / Home Page (`/`)
- Hero banner with animated badge pills and key statistics strip.
- 4-phase explanation detailing how machine learning values houses.
- Live sample valuation preview card.
- High-level model comparison summary.

### 2. Interactive Prediction Form (`/predict`)
- Clean 4-section layout grouping Quality, Spatial Dimensions, Rooms, and Age/Location.
- **1-Click Sample Presets:**
  - 🏡 *Family Suburban (Moderate)*
  - 🏰 *Luxury Executive Villa*
  - 🏠 *Starter Urban Cottage*
- Instant client-side validation and responsive helper tooltips.

### 3. Prediction Result Page (`/result`)
- **Dual Currency Valuation:** Prominently highlights price in Indian Rupees (₹) with formatted Lakhs/Crores notation alongside original US Dollars ($).
- **Documented Forex Reference:** Explicitly cites the conversion rate ($1\text{ USD} = 83.5\text{ INR}$) ensuring full transparency.
- Full tabular breakdown of submitted house specifications.

### 4. Machine Learning Dashboard (`/dashboard`)
- Live KPI cards displaying Best Model, Test $R^2$, MAE, and Dataset size.
- Interactive **Chart.js** dual-axis chart comparing $R^2$ and RMSE across all 4 models.
- Horizontal bar chart of top predictive features.
- Full model comparison table.
- High-resolution gallery of generated EDA charts.

---

## 9. Project Structure & File Guide

```
House_Price_Predict/
│
├── app.py                      # Flask backend application with routing & REST API
├── train.py                    # Complete ML training, evaluation, & plotting pipeline
├── requirements.txt            # Pinned project dependencies
├── README.md                   # Comprehensive project documentation
│
├── data/
│   └── house_prices.csv        # Ames Housing dataset (1,460 records, 81 columns)
│
├── models/
│   ├── best_model.joblib       # Serialized fitted Pipeline (preprocessor + Gradient Boosting)
│   └── model_metrics.json      # Empirical benchmark scores & metadata
│
├── notebooks/
│   └── eda.ipynb               # Jupyter Notebook documenting exploratory data analysis
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom real-estate themed styles (Navy, Slate, Emerald)
│   ├── js/
│   │   └── script.js           # Preset buttons, validation, Chart.js integrations
│   └── images/                 # Exported PNG visualizations for web display
│
├── templates/
│   ├── base.html               # Master layout with navbar, footer, and CDN links
│   ├── index.html              # Landing page and ML workflow guide
│   ├── predict.html            # Input form with presets and validations
│   ├── result.html             # Valuation card (INR & USD) and submitted specs
│   └── dashboard.html          # Performance metrics, Chart.js charts, and EDA gallery
│
└── visualizations/
    ├── price_distribution.png  # Histogram & KDE of house sale prices
    ├── correlation_heatmap.png # Pearson correlation heatmap
    ├── living_area_vs_price.png# Living area scatter plot with regression trendline
    ├── bedrooms_vs_price.png   # Bedroom count distribution box plot
    ├── neighborhood_vs_price.png # Average price per neighborhood bar chart
    └── feature_importance.png  # Top 15 feature importances from ensemble model
```

---

## 10. Step-by-Step Installation & Run Guide (macOS)

### Prerequisites
- macOS terminal (zsh / bash)
- Python 3.10 or higher installed

### Step 1: Open the Project Directory
```bash
cd /Users/ankitraj/Documents/House_Price_Predict
```

### Step 2: Activate the Virtual Environment
```bash
source .venv/bin/activate
```
*(If creating a new virtual environment from scratch: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)*

### Step 3: (Optional) Re-train and Benchmark the Models
To re-run the full training pipeline, generate fresh EDA charts, and serialize the model:
```bash
python train.py
```
*Output will display data checks, EDA export confirmations, and the 4-model comparison table.*

### Step 4: Run the Flask Web Application
```bash
python app.py
```

### Step 5: Open in Your Web Browser
Visit:
```
http://127.0.0.1:5050   # (or http://127.0.0.1:5000 if AirPlay is disabled)
```
> **Note for macOS:** macOS AirPlay Receiver uses port 5000 by default. `app.py` automatically detects this and falls back to **port 5050**.
- Navigate to **Home** (`/`) to explore the project.
- Click **Predict Price** (`/predict`) to enter custom property values or click one of the quick presets.
- Click **ML Dashboard** (`/dashboard`) to view model performance and interactive charts.

---

## 11. REST API Documentation

The application exposes a programmatic JSON endpoint for software integrations.

### Endpoint: `POST /api/predict`
- **Headers:** `Content-Type: application/json`

#### Sample Request Payload:
```json
{
  "OverallQual": 8,
  "GrLivArea": 2100,
  "BedroomAbvGr": 3,
  "FullBath": 2,
  "Floors": 2.0,
  "GarageCars": 2,
  "YearBuilt": 2010,
  "TotalBsmtSF": 1200,
  "LotArea": 11000,
  "OverallCond": 5,
  "Neighborhood": "CollgCr"
}
```

#### Sample Response:
```json
{
  "status": "success",
  "predicted_price_usd": 248645.20,
  "predicted_price_inr": 20761874.20,
  "formatted_inr": "₹ 2.08 Cr",
  "exchange_rate": 83.5,
  "model_used": "Gradient Boosting Regressor"
}
```

#### Test via cURL:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"OverallQual":8,"GrLivArea":2100,"BedroomAbvGr":3,"FullBath":2,"Floors":2.0,"GarageCars":2,"YearBuilt":2010,"TotalBsmtSF":1200,"LotArea":11000,"OverallCond":5,"Neighborhood":"CollgCr"}'
```

---

## 12. Beginner's Guide: How the Model Makes Predictions

If you are new to Machine Learning, here is a simple conceptual walkthrough of what happens under the hood:

1. **Why not just use a simple formula?**
   House prices do not follow a simple arithmetic rule. A 3-bedroom house is not simply 1.5 times more expensive than a 2-bedroom house. A house built in 2015 in a premium neighborhood with high quality finishes might be worth twice as much as an identical-sized house built in 1960 in an older neighborhood. Machine learning captures these intricate non-linear relationships.

2. **How does Gradient Boosting work?**
   Instead of building a single giant decision tree (which overfits and memorizes the data, as seen in the Decision Tree Regressor's 100% training accuracy vs 75% test accuracy), **Gradient Boosting** builds an ensemble of hundreds of small, shallow decision trees in sequence. Each new tree specifically focuses on correcting the errors (residuals) made by all prior trees.

3. **Why do we use a Scikit-Learn Pipeline?**
   Before feeding numbers into a regression algorithm, the computer needs:
   - Categorical text like `"CollgCr"` converted into binary indicator columns ($0$ or $1$) via **One-Hot Encoding**.
   - Numbers placed on comparable scales via **StandardScaler**.
   A **Pipeline** packages these transformation steps together with the model so that whatever you input on the website is preprocessed using the exact same rules learned during training.

4. **Why display both USD and INR?**
   The Ames dataset records home sales in Ames, Iowa, measured in US Dollars. Rather than pretending the US dataset was recorded in Indian rupees, our system calculates the native USD value and performs an explicit, documented conversion at $1\text{ USD} = ₹83.5$, demonstrating professional engineering integrity.

---

## 13. Future Improvements

- **Geographic Expansion:** Incorporate real estate datasets from major Indian metro areas (e.g., Bangalore, Mumbai, Delhi-NCR) with native amenities and locality indexing.
- **Advanced Hyperparameter Optimization:** Use Bayesian Optimization (`Optuna` or `GridSearchCV`) to fine-tune learning rate, subsample ratios, and tree depth.
- **Deep Learning Comparison:** Train a multi-layer perceptron (MLP) or TabNet architecture to compare deep learning tabular performance against gradient boosting.
- **Image Integration:** Allow users to upload photos of the house exterior/interior and combine computer vision features with tabular data via a multimodal architecture.

---

## 👨‍💻 Author & Portfolio Note
Developed as a Computer Science undergraduate portfolio project demonstrating practical end-to-end Machine Learning, scientific data analysis, and full-stack web development.
