# inference pipeline: predict

from pathlib import Path
import pandas as pd
import xgboost as xgb
import numpy as np
from src.feature_pipeline import rsi

DATA_PATH = Path(__file__).parent.parent / 'usdvnd' / 'processed' / 'cleaned_features_regression.csv'
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgboost_regression' / 'best_model.pkl'

def get_tomorrow_features(df_today: pd.DataFrame) -> pd.DataFrame:
    """
    Get tomorrow's features from today's data
    Args:
        df_today: pd.DataFrame: today's data
    Returns:
        pd.DataFrame: tomorrow's features
    """
    df_tomorrow = df_today[-1:].copy()
    df_tomorrow.index = df_tomorrow.index + pd.Timedelta(days=1)
    
    # Use the last known 'Lần cuối' value, not 'Target'
    last_close = df_today.iloc[-1]['Lần cuối']
    df_tomorrow['Lần cuối'] = last_close
    
    # lag features - need to look back at df_today, not shift within df_tomorrow
    # 1 day
    df_tomorrow['Lần cuối 1'] = df_today.iloc[-1]['Lần cuối']
    df_tomorrow['% Thay đổi 1'] = df_today.iloc[-1]['% Thay đổi']

    # 3 days
    df_tomorrow['Lần cuối 3'] = df_today.iloc[-3]['Lần cuối'] if len(df_today) >= 3 else last_close

    # 7 days
    df_tomorrow['Lần cuối 7'] = df_today.iloc[-7]['Lần cuối'] if len(df_today) >= 7 else last_close
    
    # sliding window features - need to calculate from df_today's recent history

    # moving average 7 and 30 days
    df_tomorrow['MA 7'] = df_today['Lần cuối'].tail(7).mean()
    df_tomorrow['MA 30'] = df_today['Lần cuối'].tail(30).mean()

    # volatility 7 days
    df_tomorrow['Std dev 7'] = df_today['Lần cuối'].tail(7).std()
    
    # RSI 14 - calculate from recent history
    df_tomorrow['RSI 14'] = rsi(df_today['Lần cuối']).iloc[-1]

    # time features
    df_tomorrow['Ngày thứ'] = df_tomorrow.index.dayofweek # 0 is Monday
    df_tomorrow['Tháng thứ'] = df_tomorrow.index.month
    df_tomorrow['Quý thứ'] = df_tomorrow.index.quarter
    df_tomorrow['Năm thứ'] = df_tomorrow.index.year
    return df_tomorrow   


async def predict_next_n_days(
    data: pd.DataFrame,
    model: xgb.XGBRegressor,
    n: int = 30,
) -> list[dict]:
    """
    Predict the next n days using the model
    Args:
        data: pd.DataFrame: data
        model: xgb.XGBRegressor: model
        n: int = 30: number of days to predict
    Returns:
        list[dict]: next n days predictions
    """

    # predict the next n days
    df = data.drop(columns=['Mở', 'Cao', 'Thấp'])
    
    for i in range(n):
        # print("Starting to get tomorrow's features")
        temp_df_tomorrow = get_tomorrow_features(df)
        # print(f"Predicting day {i+1}")
        # Predict using features (excluding Target and Lần cuối column)
        feature_cols = [col for col in temp_df_tomorrow.columns if col != 'Target' and col != 'Lần cuối']
        temp_df_tomorrow['Target'] = model.predict(temp_df_tomorrow[feature_cols])
        # print(f"Predicted Target: {temp_df_tomorrow['Target'].values[0]}")
        
        # Update 'Lần cuối' with predicted value for next iteration
        temp_df_tomorrow['Lần cuối'] = temp_df_tomorrow['Target']
        
        # print("Concatenating prediction to history")
        df = pd.concat([df, temp_df_tomorrow])
        # print(f"Latest row: {df.tail(1)[['Lần cuối', 'Target']].values}")
    
    # return index(date) and Lần cuối value of the last n days    
    df['Ngày'] = df.index
    return df.iloc[-n:][['Ngày', 'Lần cuối']].to_dict(orient='records')

if __name__ == "__main__":
    predictions = predict_next_n_days(data=data, model=model, n=30)
    print(predictions)