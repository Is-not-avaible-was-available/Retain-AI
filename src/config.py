# src/config.py

from pathlib import Path

# -------------------------
# Project paths
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_DATA_DIR = DATA_DIR / "ml"


# -------------------------
# Simulation configuration
# -------------------------

RANDOM_SEED = 42

N_CUSTOMERS = 20_000

SIMULATION_START = "2024-01-01"
SIMULATION_END = "2026-12-31"

SNAPSHOT_FREQUENCY = "MS"  # Month Start

OBSERVATION_WINDOW_DAYS = 180
PREDICTION_WINDOW_DAYS = 90

INTERVENTION_CAPACITY = 500

TARGET_ANNUAL_CHURN_RATE_LOW = 0.08
TARGET_ANNUAL_CHURN_RATE_HIGH = 0.12


# -------------------------
# Customer segments
# -------------------------

SEGMENT_DISTRIBUTION = {
    "SMB": 0.55,
    "Mid-Market": 0.30,
    "Enterprise": 0.15,
}


# -------------------------
# Industries
# -------------------------

INDUSTRIES = [
    "Technology",
    "Financial Services",
    "Manufacturing",
    "Healthcare",
    "Retail",
    "Professional Services",
    "Telecommunications",
    "Logistics",
    "Education",
    "Government",
]


# -------------------------
# Regions
# -------------------------

REGIONS = [
    "North America",
    "Europe",
    "APAC",
    "Middle East",
    "Latin America",
]


# -------------------------
# Contract types
# -------------------------

CONTRACT_TYPES = {
    "Annual": 0.65,
    "Two-year": 0.25,
    "Three-year": 0.10,
}


# -------------------------
# Products
# -------------------------

PRODUCTS = {
    "P001": "Core Platform",
    "P002": "Analytics",
    "P003": "Workflow Automation",
    "P004": "AI Assistant",
    "P005": "Advanced Reporting",
}


# -------------------------
# Output directories
# -------------------------

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ML_DATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)