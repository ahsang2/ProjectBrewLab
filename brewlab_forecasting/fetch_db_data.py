import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_database_url():
    """Return a PostgreSQL SQLAlchemy URL.
    Prefer DATABASE_URL/DB_URL. Otherwise require DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME."""
    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if db_url:
        return db_url
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required database env vars: {', '.join(missing)}")
    return "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"), db=os.getenv("DB_NAME")
    )

def fetch_data_from_db(table_name: str = "daily_sales_metrics", schema: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """
    Fetch data from PostgreSQL database and return as a pandas DataFrame.
    
    Args:
        table_name (str): Name of the table to fetch data from. Defaults to "daily_sales_metrics".
        schema (str, optional): Schema name. Defaults to None.
        limit (int, optional): Limit the number of rows returned. Defaults to None (no limit).
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the specified table.
    """
    # Create database engine
    engine = create_engine(get_database_url())
    
    # Build the query
    table_identifier = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    query = f"SELECT * FROM {table_identifier}"
    
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Execute query and fetch data
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    print(f"Fetched {len(df)} rows from {table_name}")

    return df

def fetch_data_by_date_range(start_date: str, end_date: str, table_name: str = "daily_sales_metrics", 
                           schema: str | None = None, date_column: str = "date") -> pd.DataFrame:
    """
    Fetch data from PostgreSQL database within a specific date range.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        table_name (str): Name of the table to fetch data from. Defaults to "daily_sales_metrics".
        schema (str, optional): Schema name. Defaults to None.
        date_column (str): Name of the date column. Defaults to "date".
    
    Returns:
        pd.DataFrame: DataFrame containing the data within the specified date range.
    """
    # Create database engine
    engine = create_engine(get_database_url())
    
    # Build the query
    table_identifier = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    query = f"SELECT * FROM {table_identifier} WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'"
    
    # Execute query and fetch data
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    
    print(f"Fetched {len(df)} rows from {table_name} between {start_date} and {end_date}")
    return df

if __name__ == "__main__":
    # Example usage
    try:
        # Fetch all data
        df = fetch_data_from_db()
        print(df.head())
        
        # Fetch limited data
        df_limited = fetch_data_from_db(limit=10)
        print("\nLimited data:")
        print(df_limited)
        
    except Exception as e:
        print(f"Error fetching data: {e}")