# data_processing module

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CURRENT_YEAR = pd.Timestamp.now().year


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and outliers.
    - Numeric columns: median imputation
    - Categorical columns: mode imputation
    - Remove duplicate rows
    - Outlier removal using IQR for numeric features.
    """
    df = df.copy()
    # Drop duplicates
    df = df.drop_duplicates()
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    # Impute numeric
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    # Impute categorical
    for col in cat_cols:
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode)
    # Outlier removal (IQR)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[mask]
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features.
    - House_Age = CURRENT_YEAR - YearBuilt
    - PricePerSqft = SalePrice / LotArea
    - TotalRooms = Bedrooms + Bathrooms
    - One‑hot encode Neighborhood / Location
    """
    df = df.copy()
    if "YearBuilt" in df.columns:
        df["House_Age"] = CURRENT_YEAR - df["YearBuilt"]
    if {"SalePrice", "LotArea"}.issubset(df.columns):
        df["PricePerSqft"] = df["SalePrice"] / df["LotArea"]
    if {"Bedrooms", "Bathrooms"}.issubset(df.columns):
        df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
    # One‑hot encode categorical location columns if they exist
    location_cols = [col for col in ["Neighborhood", "Location"] if col in df.columns]
    if location_cols:
        df = pd.get_dummies(df, columns=location_cols, drop_first=True)
    return df


def prepare_pipeline(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Full pipeline that returns split and scaled data ready for training.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df = feature_engineering(df)
    # Assume target column is 'SalePrice' or 'Price' depending on dataset
    target_col = "SalePrice" if "SalePrice" in df.columns else "Price"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
