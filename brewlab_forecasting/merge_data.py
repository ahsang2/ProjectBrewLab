from fetch_notion_events import fetch_notion_events
from fetch_sales_data import run_analysis
from fetch_weather_data import get_daily_weather
import pandas as pd
from datetime import datetime
import os

def merge_data(start_date, end_date):
    # Fetch data from all sources
    # Request explicit entries for all dates and normalize campaign values
    notion_events = fetch_notion_events(os.getenv("NOTION_DATABASE_ID"), start_date, end_date, os.getenv("NOTION_API_KEY"), include_all_dates=True)
    sales_data = run_analysis(start_date, end_date)
    weather_df = get_daily_weather(start_date, end_date)

    # --- Data Standardization ---

    # 1. Convert Notion events to DataFrame and standardize Date
    notion_events_df = pd.DataFrame(notion_events)
    if not notion_events_df.empty:
        notion_events_df['Date'] = pd.to_datetime(notion_events_df['Date'], format='%Y-%m-%d')
        # Normalize Campaign Type: convert NaN, empty strings and common NaN-like strings to canonical 'None'
        if 'Campaign Type' in notion_events_df.columns:
            def _norm_campaign(v):
                if pd.isna(v):
                    return 'None'
                s = str(v).strip()
                if s == '' or s.lower() in {'nan', 'n/a'}:
                    return 'None'
                return s

            notion_events_df['Campaign Type'] = notion_events_df['Campaign Type'].apply(_norm_campaign)

    # 2. Standardize Weather DataFrame
    if not weather_df.empty:
        weather_df.rename(columns={'date': 'Date'}, inplace=True)
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], format='%Y-%m-%d')

    # 3. Standardize Sales DataFrame
    grouped_sales_df = sales_data['grouped_category_sales']
    if not grouped_sales_df.empty:
        # The date format from sales data is MM/DD/YYYY
        grouped_sales_df['Date'] = pd.to_datetime(grouped_sales_df['Date'], format='%m/%d/%Y')

        # Calculate total sales using the correct column names present in the DataFrame
        sales_columns = ["Drink (Coffee)", "Drink (Non Coffee)", "Food", "Retail"]

        # Defensively ensure all required columns exist, adding them with 0 if not
        for col in sales_columns:
            if col not in grouped_sales_df.columns:
                grouped_sales_df[col] = 0.0

        grouped_sales_df['total_sales'] = grouped_sales_df[sales_columns].sum(axis=1)

    # --- Merge DataFrames ---

    # Start with the sales data as the base
    merged_df = grouped_sales_df

    # Left merge with weather data
    if not weather_df.empty:
        merged_df = pd.merge(merged_df, weather_df, on='Date', how='left')

    # Left merge with notion data
    if not notion_events_df.empty:
        merged_df = pd.merge(merged_df, notion_events_df, on='Date', how='left')

    return merged_df

if __name__ == "__main__":
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    final_df = merge_data(start, end)
    print(final_df)