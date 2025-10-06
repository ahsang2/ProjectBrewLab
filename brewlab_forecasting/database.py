import os
from datetime import datetime, timedelta
import re
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv
from merge_data import merge_data

# Load environment variables from .env file
load_dotenv()

def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalizes DataFrame columns to be valid SQL identifiers and returns a mapping."""
    df = df.copy()

    def normalize(col: str) -> str:
        s = col.strip().lower()
        s = re.sub(r'[\s\(\)]+', '_', s)  # Replace spaces, parens with underscore
        s = re.sub(r'[^a-z0-9_]+', '', s) # Remove remaining invalid chars
        s = s.strip('_')
        return s

    mapper = {col: normalize(col) for col in df.columns}
    df.rename(columns=mapper, inplace=True)

    print("Normalized column names:")
    for old, new in mapper.items():
        print(f"- '{old}' -> '{new}'")

    return df, mapper

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

def _ensure_date_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Expected a '{date_col}' column in dataframe")
    out = df.copy()
    # Keep as datetime64[ns] (so dtype mapping to SQL DATE works)
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    return out

def get_postgres_column_type(dtype):
    """Map pandas dtype to PostgreSQL column type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATE"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"

def ensure_table_with_schema_migration(engine, table_name: str, df: pd.DataFrame, schema: str | None = None, pk_col: str = "date"):
    """Create table if it doesn't exist, add missing columns if it does, and ensure primary key."""
    insp = inspect(engine)
    fq_table = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'

    if not insp.has_table(table_name, schema=schema):
        # Create new table with primary key
        print(f"Creating new table {table_name} with primary key on {pk_col}")

        # Create table structure manually to ensure primary key
        columns_sql = []
        for col in df.columns:
            col_type = get_postgres_column_type(df[col].dtype)
            if col == pk_col:
                columns_sql.append(f'"{col}" {col_type} PRIMARY KEY')
            else:
                columns_sql.append(f'"{col}" {col_type}')

        create_sql = f'CREATE TABLE {fq_table} ({", ".join(columns_sql)})'

        with engine.begin() as conn:
            conn.execute(text(create_sql))
        print(f"Table {table_name} created successfully.")
    else:
        # Table exists - check for schema drift and missing columns
        print(f"Table {table_name} exists. Checking for schema changes...")

        # Get existing columns
        existing_columns = insp.get_columns(table_name, schema=schema)
        existing_col_names = {col['name'] for col in existing_columns}
        df_columns = set(df.columns)

        # Find missing columns
        missing_columns = df_columns - existing_col_names

        if missing_columns:
            print(f"Found {len(missing_columns)} new columns. Adding to table:")
            with engine.begin() as conn:
                for col in missing_columns:
                    col_type = get_postgres_column_type(df[col].dtype)
                    alter_sql = f'ALTER TABLE {fq_table} ADD COLUMN "{col}" {col_type}'
                    print(f"  - Adding column: {col} ({col_type})")
                    conn.execute(text(alter_sql))
            print("Schema migration completed.")
        else:
            print("No schema changes needed.")

        # Ensure primary key exists
        pk_info = insp.get_pk_constraint(table_name, schema=schema)
        if not pk_info or not pk_info.get('constrained_columns'):
            print(f"Adding primary key constraint on {pk_col}")
            with engine.begin() as conn:
                # First ensure the column is NOT NULL
                conn.execute(text(f'ALTER TABLE {fq_table} ALTER COLUMN "{pk_col}" SET NOT NULL'))
                # Then add primary key constraint
                conn.execute(text(f'ALTER TABLE {fq_table} ADD PRIMARY KEY ("{pk_col}")'))
        elif pk_col not in pk_info.get('constrained_columns', []):
            print(f"Warning: Primary key exists but not on {pk_col}. Current PK: {pk_info.get('constrained_columns')}")

def upsert_df_to_postgres(df: pd.DataFrame, table_name: str, schema: str | None = None, pk_col: str = "Date"):
    if df is None or df.empty:
        print("No data to upsert.")
        return

    # Normalize column names and get mapping from original -> normalized
    df, mapper = normalize_columns(df)

    # Accept pk_col either as the original column name (e.g. "Date") or the already-normalized name (e.g. "date")
    normalized_pk_col = None

    # If user already passed a normalized column name that exists in df, use it
    if pk_col in df.columns:
        normalized_pk_col = pk_col
    else:
        # Try to find the normalized name from the mapper using the original name
        normalized_pk_col = mapper.get(pk_col)

        # Case-insensitive match against original column names if still not found
        if normalized_pk_col is None:
            for orig_col, norm_col in mapper.items():
                if str(orig_col).strip().lower() == str(pk_col).strip().lower():
                    normalized_pk_col = norm_col
                    break

    if not normalized_pk_col:
        raise ValueError(f"Primary key '{pk_col}' not found in DataFrame (tried as original and normalized names). "
                         f"Available normalized columns: {list(df.columns)}")

    # Do NOT assume the primary key is a date. Instead, try to find a real date-like column
    # (prefer original 'Date' or any of the common candidates), convert that to datetime,
    # and leave the PK column alone (it may be a composite string like 'YYYY-MM-DD_target').
    date_col = None

    # First, look for an original column named 'date' (case-insensitive) in mapper
    for orig_col, norm_col in mapper.items():
        if orig_col.strip().lower() == 'date':
            date_col = norm_col
            break

    # If not found, fallback to common normalized candidates if present in the dataframe
    if date_col is None:
        for cand in ('date', 'created_at', 'generated_at', 'timestamp'):
            if cand in df.columns:
                date_col = cand
                break

    if date_col:
        try:
            df = _ensure_date_column(df, date_col)
        except Exception as e:
            print(f"⚠️ Could not convert '{date_col}' to datetime: {e}")
    else:
        # No date-like column found; skip date normalization but warn
        print("⚠️ No date-like column found in dataframe; skipping date normalization")

    # Remove duplicates based on the primary key column to prevent ON CONFLICT errors
    original_count = len(df)
    df = df.drop_duplicates(subset=[normalized_pk_col], keep='last')
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} duplicate rows based on {normalized_pk_col}")
        
    engine = create_engine(get_database_url())

    # Use the new function with schema migration
    ensure_table_with_schema_migration(engine, table_name, df, schema=schema, pk_col=normalized_pk_col)

    md = MetaData()
    table = Table(table_name, md, schema=schema, autoload_with=engine)
    records = df.to_dict(orient="records")
    ins = pg_insert(table)
    update_cols = {c.name: ins.excluded[c.name] for c in table.columns if c.name != normalized_pk_col}
    stmt = ins.on_conflict_do_update(index_elements=[table.c[normalized_pk_col]], set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt, records)
    print(f"Upserted {len(records)} rows into {table_name}.")

def load_historical_data(start_date: datetime, end_date: datetime, table_name: str = "daily_sales_metrics", schema: str | None = None):
    print(f"Loading historical data {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    final_df = merge_data(start_date, end_date)
    if final_df is not None and not final_df.empty:
        upsert_df_to_postgres(final_df, table_name, schema=schema)
    else:
        print("No historical data to load.")

def load_daily_data(date: datetime | None = None, table_name: str = "daily_sales_metrics", schema: str | None = None):
    if date is None:
        date = datetime.now().date() - timedelta(days=0)
    print(f"Loading daily data for {date}")
    final_df = merge_data(date, date)
    if final_df is not None and not final_df.empty:
        upsert_df_to_postgres(final_df, table_name, schema=schema)
    else:
        print(f"No data available for {date}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        s = datetime.strptime(sys.argv[2], "%Y-%m-%d") if len(sys.argv) > 2 else datetime(2025, 8, 1)
        e = datetime.strptime(sys.argv[3], "%Y-%m-%d") if len(sys.argv) > 3 else datetime(2025, 9, 17)
        load_historical_data(s, e)
    elif len(sys.argv) > 1 and sys.argv[1] == "daily":
        d = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else None
        load_daily_data(d)
    else:
        load_daily_data()
