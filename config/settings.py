import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/raw/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Walmart dataset schema (IMPORTANT FIX)
TIMESTAMP_COLUMN = "Date"
TARGET_COLUMN = "Weekly_Sales"

# Walmart is weekly time series
EXPECTED_FREQUENCY = "W"