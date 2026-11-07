"""
train.py - House Price Prediction Training & Evaluation Pipeline
================================================================
This script executes the complete machine learning workflow:
1. Loads and explores the housing dataset.
2. Cleans data (handles missing values, duplicates, and outliers).
3. Engineers features (maps house style to floor count, encodes categorical variables).
4. Generates and exports high-resolution EDA charts.
5. Trains and evaluates 4 regression models (Linear Regression, Decision Tree,
   Random Forest, Gradient Boosting) using 80:20 train/test split.
6. Evaluates using R2 Score, MAE, MSE, and RMSE.
7. Automatically selects and saves the best model pipeline and metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless visualization generation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set aesthetic styling for charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Feature definitions
NUMERIC_FEATURES = [
    'OverallQual',     # Overall material and finish quality (1-10)
    'GrLivArea',        # Above grade (ground) living area sq ft
    'BedroomAbvGr',    # Bedrooms above grade
    'FullBath',        # Full bathrooms
    'Floors',          # Number of floors (e.g., 1.0, 1.5, 2.0, 2.5)
    'GarageCars',      # Garage capacity (cars)
    'YearBuilt',       # Year of construction
    'TotalBsmtSF',     # Total basement sq ft
    'LotArea',         # Lot size sq ft
    'OverallCond'      # Overall condition rating (1-10)
]

CATEGORICAL_FEATURES = [
    'Neighborhood'     # Physical location in Ames
]

TARGET = 'SalePrice'


def load_and_explore_data(csv_path: str) -> pd.DataFrame:
    """Load dataset and perform exploratory checks."""
    print("=" * 80)
    print("1. LOADING AND EXPLORING DATASET")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate records found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")

    # Map HouseStyle to numeric Floors
    style_to_floors = {
        '1Story': 1.0,
        '1.5Fin': 1.5,
        '1.5Unf': 1.5,
        '2Story': 2.0,
        '2.5Fin': 2.5,
        '2.5Unf': 2.5,
        'SFoyer': 1.0,
        'SLvl': 1.5
    }
    df['Floors'] = df['HouseStyle'].map(style_to_floors).fillna(1.0)

    # Check missing values in selected features
    all_selected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = df[all_selected].isnull().sum()
    print("\nMissing values in target and key features:")
    print(missing[missing > 0] if (missing > 0).any() else "No missing values found!")

    # Authoritative Ames outlier handling:
    # Dean De Cock (Ames Housing author) noted GrLivArea > 4000 sq ft with low price (<$300,000)
    # are atypical non-arm's length sales that distort regression lines.
    initial_count = len(df)
    outlier_mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
    outliers_count = outlier_mask.sum()
    if outliers_count > 0:
        df = df[~outlier_mask].copy()
        print(f"\nRemoved {outliers_count} extreme outlier(s) (GrLivArea > 4000 sq ft & SalePrice < $300k).")
        print(f"Rows remaining: {len(df)} (from {initial_count})")

    return df


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate and save comprehensive EDA figures."""
    print("\n" + "=" * 80)
    print("2. GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Price Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[TARGET], kde=True, color="#2563eb", bins=40, ax=ax)
    mean_price = df[TARGET].mean()
    median_price = df[TARGET].median()
    ax.axvline(mean_price, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:,.0f}')
    ax.axvline(median_price, color='#10b981', linestyle='-', linewidth=2, label=f'Median: ${median_price:,.0f}')
    ax.set_title("Distribution of House Sale Prices", pad=15, fontweight='bold')
    ax.set_xlabel("Sale Price ($)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter('${x:,.0f}')
    ax.legend()
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "price_distribution.png")
    fig.savefig(dist_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {dist_path}")

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_cols = NUMERIC_FEATURES + [TARGET]
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    ax.set_title("Correlation Heatmap of Key Features vs. Sale Price", pad=15, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {heatmap_path}")

    # 3. Living Area vs. House Price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="GrLivArea",
        y=TARGET,
        scatter_kws={'alpha': 0.5, 'color': '#0284c7'},
        line_kws={'color': '#dc2626', 'linewidth': 2},
        ax=ax
    )
    ax.set_title("Ground Living Area vs. Sale Price", pad=15, fontweight='bold')
    ax.set_xlabel("Above Ground Living Area (sq ft)")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()
    livarea_path = os.path.join(output_dir, "living_area_vs_price.png")
    fig.savefig(livarea_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {livarea_path}")

    # 4. Bedrooms vs. House Price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="BedroomAbvGr",
        y=TARGET,
        hue="BedroomAbvGr",
        legend=False,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Number of Bedrooms vs. House Sale Price", pad=15, fontweight='bold')
    ax.set_xlabel("Number of Bedrooms (Above Ground)")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()
    bed_path = os.path.join(output_dir, "bedrooms_vs_price.png")
    fig.savefig(bed_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {bed_path}")

    # 5. Neighborhood vs. Average House Price
    fig, ax = plt.subplots(figsize=(12, 8))
    neighborhood_avg = (
        df.groupby("Neighborhood")[TARGET]
        .mean()
        .reset_index()
        .sort_values(by=TARGET, ascending=True)
    )
    sns.barplot(
        data=neighborhood_avg,
        y="Neighborhood",
        x=TARGET,
        hue="Neighborhood",
        legend=False,
        palette="Spectral",
        ax=ax
    )
    ax.set_title("Average House Price by Neighborhood (Ames, Iowa)", pad=15, fontweight='bold')
    ax.set_xlabel("Average Sale Price ($)")
    ax.set_ylabel("Neighborhood")
    ax.xaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()
    neigh_path = os.path.join(output_dir, "neighborhood_vs_price.png")
    fig.savefig(neigh_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {neigh_path}")


def save_feature_importance(model, feature_names: list, output_dir: str):
    """Plot and save feature importance for tree/ensemble models."""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=fi_df,
        x='Importance',
        y='Feature',
        hue='Feature',
        legend=False,
        palette='mako',
        ax=ax
    )
    ax.set_title("Top 15 Most Important Features Influencing House Price", pad=15, fontweight='bold')
    ax.set_xlabel("Relative Feature Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(fi_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {fi_path}")


def build_preprocessor() -> ColumnTransformer:
    """
    Construct leakage-free preprocessing pipeline.
    Numerical: Impute with median, scale with StandardScaler.
    Categorical: Impute with most frequent, encode with OneHotEncoder.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train all 4 regression models and collect performance metrics."""
    print("\n" + "=" * 80)
    print("3. MODEL TRAINING & COMPARATIVE EVALUATION")
    print("=" * 80)

    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit ONLY on training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Metrics on Test Split (unseen data)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)

        # Metrics on Train Split (for overfitting diagnostic)
        r2_train = r2_score(y_train, y_train_pred)

        results[name] = {
            'train_r2': float(r2_train),
            'test_r2': float(r2_test),
            'mae': float(mae_test),
            'mse': float(mse_test),
            'rmse': float(rmse_test)
        }

    # Print Comparison Table
    print("\n" + "-" * 95)
    print(f"{'Model':<30} | {'Test R²':<10} | {'Train R²':<10} | {'MAE ($)':<12} | {'RMSE ($)':<12}")
    print("-" * 95)
    for name, m in results.items():
        print(f"{name:<30} | {m['test_r2']:<10.4f} | {m['train_r2']:<10.4f} | ${m['mae']:<11,.0f} | ${m['rmse']:<11,.0f}")
    print("-" * 95)

    # Select Best Model based on Test R2 (and minimum RMSE)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    print(f"\n★ AUTOMATIC SELECTION: Best Performing Model is '{best_model_name}' (Test R² = {results[best_model_name]['test_r2']:.4f})")

    return best_model_name, fitted_pipelines, results


def main():
    csv_path = "data/house_prices.csv"
    vis_dir = "visualizations"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load and Clean
    df = load_and_explore_data(csv_path)

    # 2. Generate Visualizations
    generate_visualizations(df, vis_dir)

    # 3. Train-Test Split (80:20)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # 4. Preprocessing & Model Benchmark
    preprocessor = build_preprocessor()
    best_name, fitted_pipelines, metrics = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    best_pipeline = fitted_pipelines[best_name]

    # 5. Extract Feature Names and Feature Importance
    fitted_preprocessor = best_pipeline.named_steps['preprocessor']
    onehot_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(onehot_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    regressor = best_pipeline.named_steps['regressor']
    save_feature_importance(regressor, all_feature_names, vis_dir)

    # 6. Save Model and Metadata
    model_save_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\n✓ Best model pipeline saved to: {model_save_path}")

    # Prepare list of unique neighborhoods sorted for the UI dropdown
    neighborhoods = sorted(df['Neighborhood'].unique().tolist())

    # Calculate top feature importance values for the dashboard
    top_features = []
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:10]
        for idx in sorted_indices:
            top_features.append({
                'feature': all_feature_names[idx],
                'importance': float(importances[idx])
            })

    metadata = {
        'best_model_name': best_name,
        'metrics': metrics,
        'dataset_summary': {
            'total_records': int(len(df)),
            'train_records': int(len(X_train)),
            'test_records': int(len(X_test)),
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'target': TARGET,
            'neighborhoods': neighborhoods
        },
        'top_features': top_features,
        'currency_conversion': {
            'usd_to_inr_rate': 83.5,
            'note': '1 USD = ₹83.5 (standard foreign exchange reference)'
        }
    }

    metrics_save_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Model metrics and metadata saved to: {metrics_save_path}")

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
