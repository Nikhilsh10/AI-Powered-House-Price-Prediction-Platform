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
    # Outlier removal (IQR) on numeric columns (excluding target if needed, but let's do it on features)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[mask]
    
    # [Step 2] Remove price-per-sqft outliers (3 standard deviations)
    if "price" in df.columns and "total_sqft" in df.columns:
        # total_sqft might be a string with ranges, so coerce to numeric first
        def parse_sqft(x):
            try:
                if isinstance(x, str) and '-' in x:
                    parts = x.split('-')
                    return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                return float(x)
            except:
                return np.nan
        df['total_sqft_num'] = df['total_sqft'].apply(parse_sqft)
        df = df.dropna(subset=['total_sqft_num'])
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft_num']  # price is in Lakhs (100,000)
        pps_mean = df['price_per_sqft'].mean()
        pps_std = df['price_per_sqft'].std()
        df = df[(df['price_per_sqft'] >= pps_mean - 3 * pps_std) & (df['price_per_sqft'] <= pps_mean + 3 * pps_std)]
        df = df.drop(columns=['total_sqft_num', 'price_per_sqft'])
        
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
    # [Step 3] Group sparse locations
    location_col = next((col for col in ["location", "Location", "Neighborhood"] if col in df.columns), None)
    if location_col:
        # Strip whitespace
        df[location_col] = df[location_col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
        location_stats = df[location_col].value_counts()
        location_stats_less_than_10 = location_stats[location_stats < 10]
        df[location_col] = df[location_col].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        
    # One‑hot encode categorical location columns if they exist
    location_cols = [col for col in ["Neighborhood", "Location", "location"] if col in df.columns]
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
