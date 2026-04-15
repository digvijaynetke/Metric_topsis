import math
import os

import numpy as np
import pandas as pd

# CONFIG: update these when the input file or columns change.
FILE_PATH = "/home/ryzen/Videos/jayesh/Uas_3hr_ACCESS-CM2_ssp126_r1i1p1f1_2015-2100 combined_output Gopalpur.xlsx"
SHEETS_TO_USE = None  # None = all sheets except the raw grid sheet (first one)
VALUE_COL = "uas"
DATE_COL = "date"
TIME_COL = "time"
TEST_FRACTION = 0.2
SU_BINS = "sqrt"  # int, or "sqrt" for sqrt(n) bins
OUTPUT_DIR = "/home/ryzen/Videos/jayesh/outputs"


def _entropy_from_hist(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def symmetric_uncertainty(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    hist2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hx = np.histogram(x, bins=x_edges)[0]
    hy = np.histogram(y, bins=y_edges)[0]
    h_x = _entropy_from_hist(hx)
    h_y = _entropy_from_hist(hy)
    h_xy = _entropy_from_hist(hist2d)
    denom = h_x + h_y
    if denom == 0:
        return float("nan")
    mi = h_x + h_y - h_xy
    return float(2.0 * mi / denom)


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.std(a, ddof=1) == 0 or np.std(b, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    rmse = math.sqrt(float(np.mean((a - b) ** 2)))
    denom = float(np.mean(a))
    if denom == 0:
        return float("nan")
    return rmse / denom


def _bins_from_rule(n: int) -> int:
    if isinstance(SU_BINS, int):
        return max(2, SU_BINS)
    if SU_BINS == "sqrt":
        return max(2, int(math.sqrt(n)))
    return max(2, int(math.sqrt(n)))


def _combine_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL in df.columns and TIME_COL in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(
            df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str), errors="coerce"
        )
        return df.sort_values("timestamp")
    return df


def _predict_persistence(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # One-step-ahead persistence forecast.
    y_true = series[1:]
    y_pred = series[:-1]
    return y_pred, y_true


def topsis(
    df: pd.DataFrame,
    benefit_cols: list[str],
    cost_cols: list[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    criteria = benefit_cols + cost_cols
    data = df[criteria].astype(float).to_numpy()

    # Vector normalization.
    norm = np.sqrt((data ** 2).sum(axis=0))
    norm[norm == 0] = np.nan
    y = data / norm

    if weights is None:
        weights = {col: 1.0 for col in criteria}

    w = np.array([weights.get(col, 1.0) for col in criteria], dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()

    p = y * w

    ideal_pos = np.zeros(p.shape[1])
    ideal_neg = np.zeros(p.shape[1])

    for j, col in enumerate(criteria):
        if col in benefit_cols:
            ideal_pos[j] = np.nanmax(p[:, j])
            ideal_neg[j] = np.nanmin(p[:, j])
        else:
            ideal_pos[j] = np.nanmin(p[:, j])
            ideal_neg[j] = np.nanmax(p[:, j])

    d_pos = np.sqrt(((p - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((p - ideal_neg) ** 2).sum(axis=1))

    denom = d_pos + d_neg
    closeness = np.where(denom == 0, np.nan, d_neg / denom)

    out = df.copy()
    out["topsis_closeness"] = closeness
    out["topsis_rank"] = out["topsis_closeness"].rank(ascending=False, method="min")
    return out


def main() -> None:
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Input file not found: {FILE_PATH}")

    xl = pd.ExcelFile(FILE_PATH)
    sheets = xl.sheet_names
    if SHEETS_TO_USE is None:
        sheets = sheets[1:]
    else:
        sheets = SHEETS_TO_USE

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for sheet in sheets:
        df = pd.read_excel(FILE_PATH, sheet_name=sheet)
        df = _combine_timestamp(df)

        if VALUE_COL not in df.columns:
            continue

        series = pd.to_numeric(df[VALUE_COL], errors="coerce").dropna().to_numpy()
        if len(series) < 3:
            continue

        y_pred, y_true = _predict_persistence(series)
        n = len(y_true)
        split = max(1, int(n * (1 - TEST_FRACTION)))
        y_true_test = y_true[split:]
        y_pred_test = y_pred[split:]

        if len(y_true_test) < 2:
            continue

        bins = _bins_from_rule(len(y_true_test))
        su = symmetric_uncertainty(y_pred_test, y_true_test, bins)
        cc = correlation_coefficient(y_pred_test, y_true_test)
        mae_val = mae(y_pred_test, y_true_test)
        nrmse_val = nrmse(y_pred_test, y_true_test)

        results.append(
            {
                "sheet": sheet,
                "su": su,
                "cc": cc,
                "mae": mae_val,
                "nrmse": nrmse_val,
            }
        )

        # Save prediction preview for traceability.
        out_path = os.path.join(OUTPUT_DIR, f"{sheet}_predictions.csv")
        out_df = pd.DataFrame(
            {
                "pred": y_pred_test,
                "obs": y_true_test,
                "err": y_pred_test - y_true_test,
            }
        )
        out_df.to_csv(out_path, index=False)

    if not results:
        print("No results computed.")
        return

    metrics_df = pd.DataFrame(results)
    print("Statistical performance indicators")
    print("(a) Symmetric Uncertainty (SU)")
    print("(b) The correlation coefficient (CC)")
    print("(c) Mean Absolute Error (MAE)")
    print("(d) Normalized root means square error (NRMSE)")
    print(metrics_df[["sheet", "su", "cc", "mae", "nrmse"]].to_string(index=False))

    benefit_cols = ["su", "cc"]
    cost_cols = ["mae", "nrmse"]
    topsis_df = topsis(metrics_df, benefit_cols, cost_cols)

    print("\nTOPSIS RESULTS")
    print(
        topsis_df[
            ["sheet", "topsis_closeness", "topsis_rank", "su", "cc", "mae", "nrmse"]
        ].sort_values("topsis_rank").to_string(index=False)
    )


if __name__ == "__main__":
    main()
