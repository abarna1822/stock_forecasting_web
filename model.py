import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def train_and_predict(csv_file, forecast_days=10):
    df = pd.read_csv(csv_file, encoding='utf-8', encoding_errors='ignore')

    # Detect 'sales' column
    original_cols = df.columns.tolist()
    lower_cols = [col.strip().lower() for col in original_cols]
    df.columns = lower_cols

    sales_keywords = ['sale', 'amount', 'revenue', 'turnover', 'earning', 'total']
    sales_col = None
    for orig, lower in zip(original_cols, lower_cols):
        if any(keyword in lower for keyword in sales_keywords):
            sales_col = lower
            break

    if not sales_col:
        raise ValueError("❌ No 'Sales' column found. Please include a column like 'Sales', 'Amount', or 'Revenue'.")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

    df = df.dropna(subset=['date'])
    df = df[['date', sales_col]]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])

    # Split into train and test
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Train Prophet model
    model = Prophet()
    model.fit(train_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=len(test_df) + forecast_days)
    forecast = model.predict(future)

    # Merge for residuals
    full_df = forecast[['ds', 'yhat']].merge(df, on='ds', how='left')
    full_df['residual'] = full_df['y'] - full_df['yhat']

    # Lag features
    residual_df = full_df.dropna(subset=['residual']).copy()
    residual_df['yhat_shift1'] = residual_df['yhat'].shift(1)
    residual_df['yhat_shift2'] = residual_df['yhat'].shift(2)
    residual_df = residual_df.dropna()

    X_train = residual_df[['yhat_shift1', 'yhat_shift2']]
    y_train = residual_df['residual']

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    # Predict on full forecast
    future_forecast = forecast.copy()
    future_forecast['yhat_shift1'] = future_forecast['yhat'].shift(1)
    future_forecast['yhat_shift2'] = future_forecast['yhat'].shift(2)
    future_forecast = future_forecast.dropna()

    xgb_preds = xgb.predict(future_forecast[['yhat_shift1', 'yhat_shift2']])

    min_len = min(len(future_forecast['yhat']), len(xgb_preds))
    future_forecast = future_forecast.tail(min_len).copy()
    future_forecast['final_yhat'] = future_forecast['yhat'].tail(min_len).values + xgb_preds[:min_len]

    eval_df = future_forecast.merge(df, on='ds', how='left')
    eval_df = eval_df.dropna(subset=['y'])

    rmse = round(np.sqrt(mean_squared_error(eval_df['y'], eval_df['final_yhat'])), 2)
    mape = round(mean_absolute_percentage_error(eval_df['y'], eval_df['final_yhat']) * 100, 2)

    # Get last N days from user input
    last_n = future_forecast.tail(forecast_days)
    predictions = list(zip(
        last_n['ds'].dt.strftime('%Y-%m-%d'),
        last_n['final_yhat'].round(2)
    ))

    historical_data = list(zip(df['ds'].dt.strftime('%Y-%m-%d'), df['y']))

    return predictions, historical_data, rmse, mape
