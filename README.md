# 🏡 House Price Prediction Using Machine Learning

An end-to-end Machine Learning web application that predicts residential property prices based on features such as living area, overall quality, location, number of bedrooms, and construction year.

Built using **Python, Scikit-learn, Flask, and Bootstrap 5**, this project covers data preprocessing, exploratory data analysis, model training, evaluation, and web deployment.

## 🚀 Key Features

- Predict house prices using user-provided property details.
- Compare four different Machine Learning regression algorithms.
- Automated data preprocessing using Scikit-learn pipelines.
- Interactive web interface built with Flask and Bootstrap.
- ML dashboard with model performance metrics and visualizations.
- Displays predicted prices in both USD ($) and INR (₹).
- REST API endpoint for programmatic predictions.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript, Bootstrap 5 |
| Interactive Charts | Chart.js |
| Model Storage | Joblib |

## 📊 Dataset

This project uses the **Ames Housing Dataset**, containing residential property information from Ames, Iowa.

- Original dataset: 1,460 records.
- Features used: 11.
- Target variable: `SalePrice`.
- Data preprocessing includes missing-value handling, outlier removal, and categorical encoding.

Important features include:
- Overall Quality
- Living Area
- Number of Bedrooms
- Full Bathrooms
- Garage Capacity
- Year Built
- Basement Area
- Lot Area
- Neighborhood

## ⚙️ Machine Learning Workflow

1. **Data Cleaning:** Handle missing values and remove extreme outliers.
2. **Exploratory Data Analysis:** Analyze price distributions, feature correlations, and relationships between house characteristics and prices.
3. **Feature Engineering:** Prepare numerical and categorical features.
4. **Preprocessing:** Use `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
5. **Model Training:** Train and compare four regression algorithms.
6. **Evaluation:** Evaluate models using R², MAE, and RMSE.
7. **Deployment:** Save the best preprocessing and prediction pipeline using Joblib and integrate it with Flask.

## 📈 Model Performance

Four regression algorithms were evaluated using an 80:20 train-test split.

| Model | Test R² Score | MAE | RMSE |
|---|---:|---:|---:|
| Gradient Boosting | **91.59%** | $15,020 | $21,549 |
| Random Forest | 88.28% | $17,619 | $25,443 |
| Linear Regression | 86.08% | $20,225 | $27,729 |
| Decision Tree | 75.17% | $27,240 | $37,032 |

**Best Performing Model: Gradient Boosting Regressor**

The model achieved a test R² score of 91.59%, making it the best-performing algorithm among the four evaluated models.

## 🖥️ Web Application

The Flask application provides:

- Home page introducing the project and prediction process.
- Prediction form with sample property presets.
- Results page displaying estimated property prices.
- Interactive dashboard showing model comparisons and feature importance.
- REST API endpoint: `POST /api/predict`.

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── data/                 # Housing dataset
├── models/               # Trained model and metrics
├── notebooks/            # EDA notebook
├── static/               # CSS, JavaScript and images
├── templates/            # Flask HTML templates
├── visualizations/       # Generated EDA plots
│
├── app.py                # Flask application
├── train.py              # Model training pipeline
├── requirements.txt      # Dependencies
└── README.md
```

## 💻 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/Ankitraj1308/House-Price-Prediction.git

cd House-Price-Prediction
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the models (optional)**

```bash
python train.py
```

**5. Run the application**

```bash
python app.py
```

Open the application in your browser at:

```text
http://127.0.0.1:5050
```

## 🔮 Future Improvements

- Integrate real estate datasets from Indian cities.
- Perform hyperparameter tuning for better model performance.
- Experiment with advanced regression algorithms.
- Explore deep learning approaches for price prediction.

## 👨‍💻 Author

**Ankit Raj**

Computer Science and Engineering Student

[GitHub](https://github.com/Ankitraj1308) 
