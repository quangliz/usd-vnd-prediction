import pandas as pd
import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

RAW_DATA_FOLDER_PATH = PROJECT_ROOT / 'usdvnd' / 'seperated'
PROCESSED_DATA_FOLDER_PATH = PROJECT_ROOT / 'usdvnd' / 'processed'

def cleaning_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cleaning the raw seperated data
    Args:
        df: pd.DataFrame: dataframe to be cleaned
    Returns:
        pd.DataFrame: cleaned dataframe
    """
    df.sort_values(by='Ngày', key=lambda x: pd.to_datetime(x, format='%d/%m/%Y'), inplace=True)
    df.drop(columns=['KL'], inplace=True)
    
    # delete special characters
    df['% Thay đổi'] = df['% Thay đổi'].str.replace('%', '')
    for col in df.columns:
        df[col] = df[col].str.replace(',', '')
        
    # convert date column to datetime and set as index
    df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
    df = df.set_index('Ngày')
    
    # create complete date range and reindex to fill missing dates
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_days)
    
    # forward fill numerical columns, fill '% Thay đổi' with 0
    num_cols = df.columns.drop('% Thay đổi')
    df[num_cols] = df[num_cols].ffill()
    df['% Thay đổi'] = df['% Thay đổi'].fillna(0)

    # turn to numerical type
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # convert index to column
    df['Ngày'] = df.index
    return df

def rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50) # prevent division by zero
    return rsi

def feature_engineering(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature engineering the cleaned data
    Args:
        df: pd.DataFrame: cleaned dataframe to be feature engineered
    Returns:
        pd.DataFrame: cleaned and feature engineered dataframe
    """
    df.set_index('Ngày', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # lag features
    # 1 day
    df['Lần cuối 1'] = df['Lần cuối'].shift(1)
    df['% Thay đổi 1'] = df['% Thay đổi'].shift(1)

    # 3 days
    df['Lần cuối 3'] = df['Lần cuối'].shift(3)

    # 7 days
    df['Lần cuối 7'] = df['Lần cuối'].shift(7)

    # sliding window features

    # moving average 7 and 30 days
    df['MA 7'] = df['Lần cuối'].rolling(window=7).mean()
    df['MA 30'] = df['Lần cuối'].rolling(window=30).mean()

    # volatility 7 days
    df['Std dev 7'] = df['Lần cuối'].rolling(window=7).std()

    # relative strength index 14 days
    df['RSI 14'] = rsi(df['Lần cuối'])

    # time features
    df['Ngày thứ'] = df.index.dayofweek # 0 is Monday
    df['Tháng thứ'] = df.index.month
    df['Quý thứ'] = df.index.quarter
    df['Năm thứ'] = df.index.year

    # drop rows with NaN
    df = df.dropna()

    # convert index to column
    df['Ngày'] = df.index
    return df

def target_formatting(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Formatting the target column for regression
    Args:
        df: pd.DataFrame: feature and cleaned dataframe
    Returns:
        pd.DataFrame: regression formatted dataframe
    """
    # create target column for regression
    df['Target'] = df['Lần cuối'].shift(-1)
    df = df.dropna(subset=['Target'])

    return df

def split_data(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splitting the data into training and testing sets for regression
    Args:
        df: pd.DataFrame: regression formatted dataframe
        split_ratio: float = 0.8: ratio of training data to total data
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: training and testing sets for regression
    """
    # split data
    split_point = int(len(df) * split_ratio)
    X_train = df.iloc[:split_point].drop(columns=['Target', 'Ngày', 'Lần cuối', 'Mở', 'Cao', 'Thấp'])
    y_train = df.iloc[:split_point]['Target']
    X_test = df.iloc[split_point:].drop(columns=['Target', 'Ngày', 'Lần cuối', 'Mở', 'Cao', 'Thấp'])
    y_test = df.iloc[split_point:]['Target']

    return X_train, y_train, X_test, y_test

def completed_data_processing(
    raw_data_folder_path: str | Path = RAW_DATA_FOLDER_PATH,
    processed_data_folder_path: str | Path = PROCESSED_DATA_FOLDER_PATH,
) -> None:
    """
    Completed the data processing
    Args:
        raw_data_folder_path: str | Path = RAW_DATA_FOLDER_PATH: folder path to load the raw data
        processed_data_folder_path: str | Path = PROCESSED_DATA_FOLDER_PATH: folder path to save the processed data
    Returns:
        None
    """
    # Convert to Path objects if strings are passed
    raw_data_folder_path = Path(raw_data_folder_path)
    # create processed data folder if it doesn't exist
    processed_data_folder_path.mkdir(parents=True, exist_ok=True)
    processed_data_folder_path = Path(processed_data_folder_path)
    
    # concat all raw seperated data
    df = pd.concat([pd.read_csv(raw_data_folder_path / file) for file in os.listdir(raw_data_folder_path) if file.endswith('.csv')])

    df = cleaning_data(df)
    df.to_csv(processed_data_folder_path / 'cleaned.csv', index=False)
    df = feature_engineering(df)
    df.to_csv(processed_data_folder_path / 'feature_engineered.csv', index=False)
    df = target_formatting(df)
    df.to_csv(processed_data_folder_path / 'cleaned_features_regression.csv', index=False)

if __name__ == "__main__":
    completed_data_processing()