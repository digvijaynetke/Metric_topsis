# Setup

## 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Run the script

```bash
python run_metrics_topsis.py
```

## 4) Change input file or columns

Edit the config block at the top of run_metrics_topsis.py:

- FILE_PATH: path to the Excel file
- SHEETS_TO_USE: set to a list of sheet names or leave None
- VALUE_COL, DATE_COL, TIME_COL: column names
- OUTPUT_DIR: where prediction CSVs are saved
