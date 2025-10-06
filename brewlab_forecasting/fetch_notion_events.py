import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

NOTION_API_URL = "https://api.notion.com/v1/databases"
NOTION_VERSION = "2022-06-28"  # or latest version


def fetch_notion_events(database_id, start_date, end_date, notion_api_key, include_all_dates: bool = True):
    url = f"{NOTION_API_URL}/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }

    # Filter for date range 
    filter_payload = {
        "filter": {
            "property": "Launch Date",
            "date": {
                "on_or_after": start_date.strftime('%Y-%m-%d'),
                "on_or_before": end_date.strftime('%Y-%m-%d')
            }
        }
    }

    events = []
    has_more = True
    next_cursor = None

    while has_more:
        payload = {
            "filter": filter_payload["filter"]
        }
        if next_cursor:
            payload["start_cursor"] = next_cursor

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        for result in data.get("results", []):
            properties = result.get("properties", {})

            # Extract the Date property
            date_val = None
            if "Launch Date" in properties and properties["Launch Date"]["type"] == "date":
                date_val = properties["Launch Date"]["date"]["start"] if properties["Launch Date"]["date"] else None
                if date_val:
                    date_val = pd.to_datetime(date_val).strftime('%Y-%m-%d')  # <-- normalize format here
                else:
                    continue  # skip if no date
            else:
                continue  # skip if no Launch Date

            # Defensive filter: skip if date is outside range
            if not (start_date.strftime('%Y-%m-%d') <= date_val <= end_date.strftime('%Y-%m-%d')):
                continue

            # Extract the Campaign Type property (assuming it's a select or text type)
            campaign_type = None
            if "Campaign Type" in properties:   
                prop = properties["Campaign Type"]
                # Handle multi_select (your case)
                if prop["type"] == "multi_select" and prop["multi_select"]:
                    campaign_type = ", ".join([v["name"] for v in prop["multi_select"]])
                elif prop["type"] == "select" and prop["select"]:
                    campaign_type = prop["select"]["name"]
                elif prop["type"] == "title" and prop["title"]:
                    campaign_type = prop["title"][0]["plain_text"]
                elif prop["type"] == "rich_text" and prop["rich_text"]:
                    campaign_type = prop["rich_text"][0]["plain_text"]

            # Extract the Locations property (assuming it's a multi_select or select)
            locations = []
            if "Locations" in properties:
                prop = properties["Locations"]
                if prop["type"] == "multi_select" and prop["multi_select"]:
                    locations = [loc["name"] for loc in prop["multi_select"]]
                elif prop["type"] == "select" and prop["select"]:
                    locations = [prop["select"]["name"]]

           # Filter for Locations being 'All Locations', 'BrewLab', or empty
            if locations and not any(loc in ["All Locations", "BrewLab"] for loc in locations):
                continue

            # Append filtered event with only Date and Campaign Type
            # Normalize campaign_type: treat None, empty strings, and common NaN variations as the canonical label 'None'
            campaign_val = campaign_type
            if campaign_val is None:
                campaign_val = "None"
            else:
                try:
                    s = str(campaign_val).strip()
                    if s == "" or s.lower() in {"nan", "n/a"}:
                        campaign_val = "None"
                    else:
                        campaign_val = s
                except Exception:
                    campaign_val = "None"

            events.append({"Date": date_val, "Campaign Type": campaign_val})

        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")

    # If requested, ensure there's an explicit entry for every date in the requested range.
    # This prevents downstream merges from producing NaN for missing Campaign Type values.
    if include_all_dates:
        try:
            existing = {e["Date"] for e in events}
            cur = start_date
            while cur <= end_date:
                dstr = cur.strftime('%Y-%m-%d')
                if dstr not in existing:
                    events.append({"Date": dstr, "Campaign Type": "None"})
                cur = cur + timedelta(days=1)
        except Exception:
            # Be conservative: if anything goes wrong here, just return events as-is
            pass
    return events


if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Set your environment variables or replace with actual values
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_database_id = os.getenv("NOTION_DATABASE_ID")

    if not notion_api_key or not notion_database_id:
        print("Please set NOTION_API_KEY and NOTION_DATABASE_ID environment variables.")
        exit(1)

    # Parse command line arguments for date range (YYYY-MM-DD)
    if len(sys.argv) >= 3:
        start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    else:
        # Define your date range directly
        start_date = datetime(2024, 1, 1)  # January 1, 2024
        end_date = datetime(2024, 12, 31)  # December 31, 2024

    # Call the function directly
    notion_events_df = fetch_notion_events(notion_database_id, start_date, end_date, notion_api_key)
    #print(notion_events_df)

    # You can also call it with different date ranges
    # df_q1 = fetch_notion_events(notion_database_id, datetime(2024, 1, 1), datetime(2024, 3, 31), notion_api_key)
    # df_q2 = fetch_notion_events(notion_database_id, datetime(2024, 4, 1), datetime(2024, 6, 30), notion_api_key)