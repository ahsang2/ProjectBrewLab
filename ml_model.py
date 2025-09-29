"""
Sales Forecasting Model for Brewlab

This script trains machine learning models to forecast sales for different product categories
(drink_coffee, drink_non_coffee, food, retail) using historical sales data, weather data,
and campaign information.

Usage:
    python ml_model.py

Configuration:
    Modify the CONFIG dictionary below to adjust model parameters, forecast horizon,
    and other settings without diving into the code.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from database import get_database_url
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fetch_weather_data import get_daily_weather

load_dotenv()

# =============================================================================
# CONFIGURATION - Modify these settings to customize the model
# =============================================================================
CONFIG = {
    # Target columns to forecast
    'target_columns': ['drink_coffee', 'drink_non_coffee', 'food', 'retail', 'total_sales'],

    # Forecast horizon (days into the future)
    'forecast_horizon': 10,

    # Model hyperparameters
    'model_params': {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    
    # Feature engineering settings
    'lag_windows': [1, 7],  # Which lag features to create
    'rolling_windows': [7, 28],  # Which rolling average windows to create
    
    # Training settings
    'test_size': 0.2,  # Fraction of data to use for validation
    'early_stopping_rounds': 100,
    
    # Database settings
    'table_name': 'daily_sales_metrics',
    'forecast_table': 'sales_forecasts',
    
    # Plotting settings
    'figure_size': (12, 6),
    'show_plots': True
}

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data_from_db(table_name=None):
    """Load sales data from the database."""
    if table_name is None:
        table_name = CONFIG['table_name']
    
    engine = create_engine(get_database_url())
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        
        # Handle date column (could be 'date' or 'Date')
        date_col = 'date' if 'date' in df.columns else 'Date'
        if date_col not in df.columns:
            raise ValueError("No date column found. Expected 'date' or 'Date'.")
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        print(f"✅ Loaded {len(df)} rows of data from {table_name}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()


def preprocess_data(df):
    """Clean and prepare data for modeling."""
    df = df.copy()

    # Handle campaign_type column
    if 'campaign_type' in df.columns:
        df['campaign_type'] = df['campaign_type'].fillna('No Campaign')
        df = pd.get_dummies(df, columns=['campaign_type'], prefix='campaign', dummy_na=False)
        print("✅ One-hot encoded campaign_type column")

    # Ensure total_sales exists (backfill if older data/table missing the new column)
    if 'total_sales' not in df.columns:
        base_cols = ['drink_coffee', 'drink_non_coffee', 'food', 'retail']
        missing = [c for c in base_cols if c not in df.columns]
        if not missing:
            df['total_sales'] = df[base_cols].sum(axis=1)
            print("✅ Added missing total_sales column from component categories")
        else:
            print(f"⚠️  Could not create total_sales because missing components: {missing}")

    # Fill any remaining missing values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_time_features(df):
    """Create time-based features from the datetime index."""
    df = df.copy()
    
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    return df


def create_lag_features(df, target_col):
    """Create lag and rolling window features for the target column."""
    df = df.copy()
    
    # Create lag features
    for lag in CONFIG['lag_windows']:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    
    # Create rolling window features
    for window in CONFIG['rolling_windows']:
        df[f'{target_col}_roll{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).mean()
    
    return df


def build_feature_matrix(df, target_col):
    """Build the complete feature matrix for training."""
    # Start with time features
    df_features = create_time_features(df)
    
    # Add lag features for the target
    df_features = create_lag_features(df_features, target_col)
    
    # Define feature columns
    time_features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    lag_features = [f'{target_col}_lag{lag}' for lag in CONFIG['lag_windows']]
    rolling_features = [f'{target_col}_roll{window}' for window in CONFIG['rolling_windows']]
    
    # Exogenous features (all other numeric columns except target and other targets)
    # Exogenous features (all other numeric columns except target and other targets)
    exog_features = [col for col in df_features.select_dtypes(include=[np.number]).columns
                     if col != target_col and col not in CONFIG['target_columns'] and col not in time_features
                     and not any(col.endswith(f'_lag{lag}') for lag in CONFIG['lag_windows'])
                     and not any(col.endswith(f'_roll{window}') for window in CONFIG['rolling_windows'])]

    # Campaign features
    campaign_features = [col for col in df_features.columns if col.startswith('campaign_')]
    
    all_features = time_features + lag_features + rolling_features + exog_features + campaign_features
    
    # Remove rows with NaN values (due to lags)
    df_clean = df_features.dropna(subset=[target_col] + all_features)
    
    X = df_clean[all_features]
    y = df_clean[target_col]
    
    print(f"✅ Built feature matrix: {len(X)} samples, {len(all_features)} features")
    return X, y, all_features


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_model(X, y):
    """Train a LightGBM model with the given features and target."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=CONFIG['test_size'], shuffle=False
    )
    
    model = lgb.LGBMRegressor(**CONFIG['model_params'])
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False)]
    )
    
    return model, X_valid, y_valid


def evaluate_model(model, X_valid, y_valid, target_name):
    """Calculate and display model performance metrics."""
    preds = model.predict(X_valid)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_valid, preds)),
        'mae': mean_absolute_error(y_valid, preds),
        'r2': r2_score(y_valid, preds),
        'mape': mean_absolute_percentage_error(y_valid, preds)
    }
    
    print(f"\n📊 Evaluation Metrics for {target_name}")
    print("=" * 40)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}")
    
    return metrics


# =============================================================================
# FORECASTING
# =============================================================================

def get_future_weather(start_date, end_date):
    """Get weather forecast for future dates."""
    try:
        weather_df = get_daily_weather(start_date, end_date)
        if not weather_df.empty:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_df = weather_df.set_index('date')
            print(f"✅ Retrieved weather forecast for {len(weather_df)} days")
            return weather_df
    except Exception as e:
        print(f"⚠️  Could not get weather forecast: {e}")
    
    return pd.DataFrame()


def create_future_features(df, future_dates, target_col, feature_cols):
    """Create feature matrix for future dates."""
    future_df = pd.DataFrame(index=future_dates)
    
    # Add time features
    future_df = create_time_features(future_df)
    
    # Get weather forecast
    weather_forecast = get_future_weather(future_dates.min(), future_dates.max())
    
    # Add weather features if available
    weather_cols = [col for col in feature_cols if any(w in col.lower() for w in ['temp', 'precip', 'wind'])]
    for col in weather_cols:
        if col in weather_forecast.columns:
            future_df[col] = weather_forecast[col]
        else:
            # Use historical median for missing weather data
            future_df[col] = df[col].median()
    
    # Add campaign features (assume no future campaigns)
    campaign_cols = [col for col in feature_cols if col.startswith('campaign_')]
    for col in campaign_cols:
        if col == 'campaign_No Campaign':
            future_df[col] = 1
        else:
            future_df[col] = 0
    
    # Fill any remaining missing features with historical medians
    for col in feature_cols:
        if col not in future_df.columns:
            if col in df.columns:
                future_df[col] = df[col].median()
            else:
                future_df[col] = 0
    
    return future_df[feature_cols]


def forecast_sales(model, df, target_col, feature_cols, horizon=None):
    """Generate sales forecast for the specified horizon."""
    if horizon is None:
        horizon = CONFIG['forecast_horizon']
    
    last_date = df.index.max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq='D')
    
    # For recursive forecasting with lags, we need to predict one day at a time
    forecasts = []
    extended_df = df.copy()
    
    for future_date in future_dates:
        # Create features for this single future date
        single_date_index = pd.DatetimeIndex([future_date])
        future_features = create_future_features(extended_df, single_date_index, target_col, feature_cols)
        
        # Add lag features based on extended history (including previous predictions)
        for lag in CONFIG['lag_windows']:
            lag_col = f'{target_col}_lag{lag}'
            if lag_col in feature_cols:
                if len(extended_df) >= lag:
                    future_features.loc[future_date, lag_col] = extended_df[target_col].iloc[-lag]
                else:
                    future_features.loc[future_date, lag_col] = extended_df[target_col].mean()
        
        # Add rolling features
        for window in CONFIG['rolling_windows']:
            roll_col = f'{target_col}_roll{window}'
            if roll_col in feature_cols:
                recent_values = extended_df[target_col].tail(window)
                future_features.loc[future_date, roll_col] = recent_values.mean()
        
        # Make prediction
        pred = model.predict(future_features)[0]
        forecasts.append(pred)
        
        # Add prediction to extended history for next iteration
        extended_df.loc[future_date, target_col] = pred
    
    return pd.Series(forecasts, index=future_dates, name=f'{target_col}_forecast')


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_forecast(df, forecast, target_col):
    """Plot historical data and forecast."""
    if not CONFIG['show_plots']:
        return
    
    plt.figure(figsize=CONFIG['figure_size'])
    
    # Plot historical data (last 60 days for clarity)
    recent_data = df[target_col].tail(60)
    recent_data.plot(label='Historical', color='steelblue', linewidth=2)
    
    # Plot forecast
    forecast.plot(label='Forecast', color='orange', linewidth=2, marker='o')
    
    plt.title(f'{CONFIG["forecast_horizon"]}-Day Sales Forecast: {target_col.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def save_forecasts_to_db(forecasts_dict):
    """Save forecasts to the database."""
    if not forecasts_dict:
        print("⚠️  No forecasts to save")
        return

    # Combine all forecasts into a single DataFrame
    forecast_records = []
    for target, forecast_series in forecasts_dict.items():
        for date, value in forecast_series.items():
            # Round forecast_value to 2 decimal places when building records
            forecast_records.append({
                'date': date.date(),
                'target_column': target,
                'forecast_value': round(float(value), 2) if pd.notnull(value) else value,
                'created_at': datetime.now()
            })

    forecast_df = pd.DataFrame(forecast_records)

    # Ensure forecast_value is numeric and rounded to 2 decimals (safety step)
    if 'forecast_value' in forecast_df.columns:
        forecast_df['forecast_value'] = pd.to_numeric(forecast_df['forecast_value'], errors='coerce').round(2)

    # Save to database
    engine = create_engine(get_database_url())
    try:
        forecast_df.to_sql(CONFIG['forecast_table'], engine, if_exists='replace', index=False)
        print(f"✅ Saved {len(forecast_df)} forecast records to {CONFIG['forecast_table']}")
    except Exception as e:
        print(f"❌ Error saving forecasts: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🚀 Starting Sales Forecasting Pipeline")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_data_from_db()
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    df = preprocess_data(df)
    
    # Train models and generate forecasts
    all_forecasts = {}
    all_metrics = {}
    
    for target_col in CONFIG['target_columns']:
        if target_col not in df.columns:
            print(f"⚠️  Target column '{target_col}' not found in data. Skipping.")
            continue
        
        print(f"\n🎯 Processing target: {target_col}")
        print("-" * 30)
        
        # Build features
        X, y, feature_cols = build_feature_matrix(df, target_col)
        
        if len(X) < 50:
            print(f"⚠️  Not enough data for {target_col} (only {len(X)} samples). Skipping.")
            continue
        
        # Train model
        print("🔧 Training model...")
        model, X_valid, y_valid = train_model(X, y)
        
        # Evaluate model
        metrics = evaluate_model(model, X_valid, y_valid, target_col)
        all_metrics[target_col] = metrics
        
        # Generate forecast
        print(f"🔮 Generating {CONFIG['forecast_horizon']}-day forecast...")
        forecast = forecast_sales(model, df, target_col, feature_cols)
        all_forecasts[target_col] = forecast
        
        # Plot results
        plot_forecast(df, forecast, target_col)
    
    # Save forecasts to database
    save_forecasts_to_db(all_forecasts)
    
    print("\n🎉 Forecasting pipeline completed!")
    print("=" * 50)
    
    return all_forecasts, all_metrics


if __name__ == "__main__":
    forecasts, metrics = main()