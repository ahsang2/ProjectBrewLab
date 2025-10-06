# Files:

requirements.txt - requred packages to install
.env - store environment variables here (use env_template)

fetch_notion_events.py - fetches events data from Notion calendar
fetch_sales_data.py - fetches sales data from Square
fetch_weather_data.py - fetches weather data from API
merge_data.py - merges data fetched from above scripts

database.py - handles database

fetch_db_data.py - fetches data from database for ml model
model.py - ml model, backtest, forecast

main.py - runs full pipeline (database.py, model.py)


# Terminal commands:

## Install packages
```bash
pip install -r requirements.txt
```
## Run pipeline
```bash
python main.py
```
## Database ops
For todays data:
```bash
python database.py
``` 
or 
```bash
python database.py daily
```
For historical data (Replace dates after historical to desired range):
```bash
python database.py historical 2025-09-05 2025-10-05
```
## ML Model/Backtest/Forecasts
```bash
python model.py
```
# Automation

1. Make it executable (Replace /full/path/to/main.py with the actual full path to your main.py file ):
```bash
chmod +x /full/path/to/main.py
```
2. Open Terminal and type:
```bash 
crontab -e
```
3. Add this line (Runs at 6PM. Replace /path/to/your/main.py with the actual full path to your main.py file):
```bash
0 18 * * * /usr/bin/python3 /path/to/your/main.py
```
4. Save and exit
    1. press Esc
    2. type :wq
    3. press Enter
5. Check scheduled cron job:
```bash
crontab -l
```



