import os
from pathlib import Path

print('Current working directory:', os.getcwd())
print('Data exists (relative):', Path('data/processed/clean_data.csv').exists())
PROJECT_ROOT = Path(__file__).resolve().parents[3]
print('Resolved PROJECT_ROOT:', PROJECT_ROOT)
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'clean_data.csv'
METRICS_PATH = PROJECT_ROOT / 'models' / 'metrics.json'
print('DATA_PATH:', DATA_PATH)
print('DATA_PATH exists:', DATA_PATH.exists())
print('METRICS_PATH:', METRICS_PATH)
print('METRICS_PATH exists:', METRICS_PATH.exists())
