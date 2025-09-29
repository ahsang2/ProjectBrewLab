# fetch_sales_data.py

import os
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID")

if not SQUARE_ACCESS_TOKEN:
    raise EnvironmentError("Missing SQUARE_ACCESS_TOKEN in .env file")

if not SQUARE_LOCATION_ID:
    raise EnvironmentError("Missing SQUARE_LOCATION_ID in .env file")

HEADERS = {
    "Square-Version": "2024-06-12",
    "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

# ============================================================================
# SECTION 1: Extract Square Categories (from extract_square_categories.py)
# ============================================================================

def fetch_square_catalog(types):
    """Fetch catalog objects from Square API"""
    url = "https://connect.squareup.com/v2/catalog/list"
    params = {"types": types}

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch catalog objects: {response.text}")

    return response.json().get("objects", [])

def fetch_square_items():
    """Fetch all items from Square catalog"""
    return fetch_square_catalog("ITEM")

def fetch_square_categories():
    """Fetch all categories from Square catalog"""
    return fetch_square_catalog("CATEGORY")

def extract_item_categories():
    """Extract item-category mappings from Square"""
    print("📋 Extracting item categories from Square...")
    
    items = fetch_square_items()
    categories = fetch_square_categories()

    # Build a map of category_id to category_name
    category_map = {cat["id"]: cat["category_data"]["name"] for cat in categories if cat.get("type") == "CATEGORY"}

    item_category_list = []

    for item in items:
        if item.get("type") != "ITEM":
            continue

        item_id = item.get("id")
        item_data = item.get("item_data", {})
        item_name = item_data.get("name", "Unknown")

        # Attempt to extract category_id from reporting_category or categories[0]
        reporting_category = item_data.get("reporting_category", {}).get("id")
        categories_list = item_data.get("categories", [])
        fallback_category = categories_list[0].get("id") if categories_list else None

        category_id = reporting_category or fallback_category
        category_name = category_map.get(category_id, "Uncategorized")

        item_category_list.append({
            "item_id": item_id,
            "item_name": item_name,
            "category_id": category_id or "None",
            "category_name": category_name
        })

    print(f"✅ Extracted {len(item_category_list)} item-category mappings")
    return item_category_list

# ============================================================================
# SECTION 2: Get Item Sales (from get_item_sales.py)
# ============================================================================

def fetch_orders(start_date: datetime, end_date: datetime):
    """Fetch orders from Square API within date range"""
    url = "https://connect.squareup.com/v2/orders/search"
    all_orders = []
    cursor = None

    while True:
        body = {
            "location_ids": [SQUARE_LOCATION_ID],
            "query": {
                "filter": {
                    "date_time_filter": {
                        "created_at": {
                            "start_at": start_date.isoformat(),
                            "end_at": end_date.isoformat()
                        }
                    },
                    "state_filter": {
                        "states": ["COMPLETED"]
                    }
                }
            },
            "limit": 500
        }
        if cursor:
            body["cursor"] = cursor

        response = requests.post(url, headers=HEADERS, json=body)
        response.raise_for_status()
        data = response.json()

        all_orders.extend(data.get("orders", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    return all_orders
def get_daily_sales_by_item(start_date: datetime, end_date: datetime):
    """Get daily sales data by item"""
    # Be tolerant of either datetime.datetime or datetime.date for inputs:
    # If the object has a .date() method (i.e., datetime), convert to date for display.
    start_date_obj = start_date.date() if hasattr(start_date, "date") else start_date
    end_date_obj = end_date.date() if hasattr(end_date, "date") else end_date

    print(f"📊 Fetching sales data from {start_date_obj} to {end_date_obj}...")

    orders = fetch_orders(start_date, end_date)
    if not orders:
        print("No orders found for the given period.")
        # Return an empty dataframe with an Item column to keep downstream logic simple
        empty_df = pd.DataFrame(columns=["Item"])
        print("✅ Retrieved sales data for 0 items")
        return empty_df

    sales_by_item = defaultdict(lambda: defaultdict(float))

    for order in orders:
        created_at = order.get("created_at")
        if not created_at:
            continue

        try:
            date_key = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%m/%d/%Y")
        except Exception:
            # Fallback: try a simple split (ISO-like) if parsing fails
            try:
                date_key = created_at.split("T")[0]
            except Exception:
                continue

        for item in order.get("line_items", []):
            name = item.get("name", "Unknown")
            amount = item.get("total_money", {}).get("amount", 0) / 100.0
            sales_by_item[name][date_key] += amount

    # Create a DataFrame
    df = pd.DataFrame(sales_by_item).T.fillna(0)
    df.index.name = "Item"
    df = df.reset_index()
    df = df[["Item"] + sorted([col for col in df.columns if col != "Item"])]

    print(f"✅ Retrieved sales data for {len(df)} items")
    return df

# ============================================================================
# SECTION 3: Combine Item Category Sales (from combine_item_category_sales.py)
# ============================================================================

def combine_item_category_sales(sales_df, category_list):
    """Combine sales data with category information"""
    print("🔗 Combining item sales with category data...")

    # Convert category list to DataFrame
    category_df = pd.DataFrame(category_list)

    # Rename columns to align for merging
    sales_df_copy = sales_df.copy()
    category_df_renamed = category_df.rename(columns={"item_name": "Item", "category_name": "Category Name"})

    # Merge to add category info to each item
    merged_df = pd.merge(sales_df_copy, category_df_renamed[["Item", "Category Name"]], on="Item", how="left")

    # Keep only sales columns + necessary ID vars
    sales_columns = [col for col in merged_df.columns if col.startswith("0") or col.startswith("1") or "/" in col]
    melted_df = merged_df[["Item", "Category Name"] + sales_columns].melt(
        id_vars=["Item", "Category Name"],
        var_name="Date",
        value_name="Sales"
    )

    # Group by Category and Date
    grouped = (
        melted_df.groupby(["Date", "Category Name"])["Sales"]
        .sum()
        .reset_index()
    )

    # Pivot to make Categories into columns
    pivot_df = grouped.pivot(index="Date", columns="Category Name", values="Sales").fillna(0)

    # Reorder columns alphabetically (optional)
    pivot_df = pivot_df[sorted(pivot_df.columns)]
    pivot_df = pivot_df.reset_index()

    print(f"✅ Created category sales data with {len(pivot_df.columns)-1} categories")
    return pivot_df

# ============================================================================
# SECTION 4: Map Category Groups (from map_category_groups.py)
# ============================================================================

def map_category_groups(category_sales_df):
    """Map categories to broader groups"""
    print("📊 Mapping categories to broader groups...")

    # Define mapping from category to broader group
    category_to_group = {
        "Coffee": "Drink (Coffee)",
        "Signature Drinks": "Drink (Coffee)",
        "Espresso Drinks": "Drink (Coffee)",
        "Draft": "Drink (Coffee)",

        "Handcrafted Drinks": "Drink (Non Coffee)",
        "Teas": "Drink (Non Coffee)",
        "Staff Picks": "Drink (Non Coffee)",
        "Refreshers": "Drink (Non Coffee)",
        "Other Drinks": "Drink (Non Coffee)",

        "Pastry": "Food",
        "BakeLab": "Food",

        "ACC": "IGNORE",
        "Uncategorized": "IGNORE",

        "RTD": "Retail",
        "Beans": "Retail",
        "Loopy Dopamine": "Retail",
        "Hario": "Retail",
        "Pottery": "Retail",
        "Framed": "Retail",
        "Chen's Crochet": "Retail",
        "Planted": "Retail",
        "Brewlab Retail": "Retail",
        "Caring Candle": "Retail",
        "Brew Bottled Drinks": "Retail",
    }

    # Initialize a new DataFrame for grouped results
    grouped_df = pd.DataFrame()
    grouped_df["Date"] = category_sales_df["Date"]

    # Aggregate sales for each broader group
    for group in set(category_to_group.values()):
        matching_columns = [cat for cat, grp in category_to_group.items() if grp == group and cat in category_sales_df.columns]
        if matching_columns:
            grouped_df[group] = category_sales_df[matching_columns].sum(axis=1)
        else:
            grouped_df[group] = 0.0

    print(f"✅ Created grouped sales data with {len(grouped_df.columns)-1} groups")
    return grouped_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis(start_date=None, end_date=None):
    """
    Run the complete Square sales analysis and return all dataframes

    Args:
        start_date (datetime, optional): Start date for analysis
        end_date (datetime, optional): End date for analysis

    Returns:
        dict: Dictionary containing all generated dataframes
    """
    print("🚀 Starting Square Sales Analysis Pipeline")
    print("=" * 50)

    # Use provided dates or sensible defaults (now)
    if start_date is None:
        start_date = datetime.now()
    if end_date is None:
        end_date = datetime.now()

    # Normalize date inputs: convert date -> datetime (start at midnight, end at end of day)
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())

    # If user accidentally passed start > end, swap and warn
    if start_date > end_date:
        print("⚠️  Warning: start_date is after end_date. Swapping the values.")
        start_date, end_date = end_date, start_date

    try:
        # Step 1: Extract item categories from Square
        item_categories = extract_item_categories()
        item_category_df = pd.DataFrame(item_categories)

        # Step 2: Get daily sales by item
        print(f"📊 Fetching sales data from {start_date.date()} to {end_date.date()}...")
        sales_df = get_daily_sales_by_item(start_date, end_date)

        # Step 3: Combine item sales with category data
        category_sales_df = combine_item_category_sales(sales_df, item_categories)

        # Step 4: Map categories to broader groups
        grouped_sales_df = map_category_groups(category_sales_df)

        print("\n" + "=" * 50)
        print("✅ Analysis Complete!")

        # Display summary of final results
        print(f"\n📈 Final Summary:")
        print(f"   • Date range: {start_date.date()} to {end_date.date()}")
        try:
            items_count = len(sales_df)
        except Exception:
            items_count = 0
        # Safely compute category and group counts (subtracting one if the first column is an index/date column)
        def safe_column_count(df):
            cols = getattr(df, "columns", None)
            if cols is None:
                return 0
            return max(0, len(cols) - 1)

        categories_count = safe_column_count(category_sales_df)
        groups_count = safe_column_count(grouped_sales_df)

        print(f"   • Items analyzed: {items_count}")
        print(f"   • Categories: {categories_count}")
        print(f"   • Groups: {groups_count}")

        # Return all dataframes in a dictionary
        return {
            'item_categories': item_category_df,
            'daily_sales_by_item': sales_df,
            'daily_sales_by_category': category_sales_df,
            'grouped_category_sales': grouped_sales_df
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    print("sys.argv:", sys.argv)  # Debugging line
    from datetime import datetime

    # Parse command line arguments for date range (YYYY-MM-DD)
    if len(sys.argv) >= 3:
        start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
        print(f"Parsed start: {start}, end: {end}")  # Debugging line
    else:
        # Defaults (same as previously used defaults)
        start = datetime(2024, 12, 8)
        end = datetime(2024, 12, 20)

    print(f"Running sales analysis from {start.date()} to {end.date()}")
    results = run_analysis(start, end)

    # Display a sample of the final results
    if results:
        print("\n📊 Sample of grouped sales data:")
        print(results['grouped_category_sales'].head())
