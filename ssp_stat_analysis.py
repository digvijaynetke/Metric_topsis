import itertools
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CONFIG
SSP_DIR = os.path.join(os.path.dirname(__file__), "ssp_files")
COLUMN_NAME = "uas"
SHEET_NAME = 0  # First sheet
MIN_ROWS = 2
FLOAT_FMT = "{:.6f}"
ESM_DIR = os.path.join(os.path.dirname(__file__), "ESM")
ESM_OUTPUT = os.path.join(ESM_DIR, "ssp_stat_results.xlsx")
IMG_JOINT_DIR = os.path.join(os.path.dirname(__file__), "img_joint_pdf")
IMG_NORMAL_DIR = os.path.join(os.path.dirname(__file__), "img_normal_dist")
HIST_BINS = 50


def _find_column(df: pd.DataFrame) -> str | None:
    if COLUMN_NAME in df.columns:
        return COLUMN_NAME
    lower_map = {str(col).lower(): col for col in df.columns}
    return lower_map.get(COLUMN_NAME.lower())


def _mode_value(series: pd.Series) -> float:
    modes = series.mode(dropna=True)
    if modes.empty:
        return float("nan")
    return float(modes.iloc[0])


def _load_series(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    col = _find_column(df)
    if col is None:
        raise ValueError(f"Column '{COLUMN_NAME}' not found in {os.path.basename(file_path)}")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return series.reset_index(drop=True)


def _series_stats(series: pd.Series) -> dict[str, float]:
    return {
        "count": float(series.size),
        "mean": float(series.mean()) if series.size else float("nan"),
        "mode": _mode_value(series),
        "median": float(series.median()) if series.size else float("nan"),
        "std": float(series.std(ddof=1)) if series.size > 1 else float("nan"),
        "kurtosis": float(series.kurtosis()) if series.size else float("nan"),
        "skewness": float(series.skew()) if series.size else float("nan"),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.std(a, ddof=1) == 0 or np.std(b, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pair_metrics(a: pd.Series, b: pd.Series) -> dict[str, float]:
    n = min(len(a), len(b))
    if n < MIN_ROWS:
        return {
            "aligned_n": float(n),
            "correlation_r": float("nan"),
            "rmse": float("nan"),
            "mean_error": float("nan"),
        }

    a_vals = a.iloc[:n].to_numpy()
    b_vals = b.iloc[:n].to_numpy()
    diff = a_vals - b_vals

    return {
        "aligned_n": float(n),
        "correlation_r": _correlation(a_vals, b_vals),
        "rmse": math.sqrt(float(np.mean(diff ** 2))),
        "mean_error": float(np.mean(diff)),
    }


def _format_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return FLOAT_FMT.format(float(value))
    return str(value)


def _build_table(
    name_a: str,
    name_b: str,
    stats_a: dict[str, float],
    stats_b: dict[str, float],
    pair: dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "metric": "count",
            "file_a": _format_value(stats_a["count"]),
            "file_b": _format_value(stats_b["count"]),
            "pair": _format_value(pair["aligned_n"]),
        },
        {
            "metric": "mean",
            "file_a": _format_value(stats_a["mean"]),
            "file_b": _format_value(stats_b["mean"]),
            "pair": "",
        },
        {
            "metric": "mode",
            "file_a": _format_value(stats_a["mode"]),
            "file_b": _format_value(stats_b["mode"]),
            "pair": "",
        },
        {
            "metric": "median",
            "file_a": _format_value(stats_a["median"]),
            "file_b": _format_value(stats_b["median"]),
            "pair": "",
        },
        {
            "metric": "std",
            "file_a": _format_value(stats_a["std"]),
            "file_b": _format_value(stats_b["std"]),
            "pair": "",
        },
        {
            "metric": "kurtosis",
            "file_a": _format_value(stats_a["kurtosis"]),
            "file_b": _format_value(stats_b["kurtosis"]),
            "pair": "",
        },
        {
            "metric": "skewness",
            "file_a": _format_value(stats_a["skewness"]),
            "file_b": _format_value(stats_b["skewness"]),
            "pair": "",
        },
        {
            "metric": "correlation_r",
            "file_a": "",
            "file_b": "",
            "pair": _format_value(pair["correlation_r"]),
        },
        {
            "metric": "rmse",
            "file_a": "",
            "file_b": "",
            "pair": _format_value(pair["rmse"]),
        },
        {
            "metric": "mean_error",
            "file_a": "",
            "file_b": "",
            "pair": _format_value(pair["mean_error"]),
        },
    ]

    return pd.DataFrame(rows)


def _block_for_excel(name_a: str, name_b: str, table: pd.DataFrame) -> pd.DataFrame:
    pair_row = {
        "metric": "Pair",
        "file_a": f"{name_a} vs {name_b}",
        "file_b": "",
        "pair": "",
    }
    blank_row = {"metric": "", "file_a": "", "file_b": "", "pair": ""}
    block = pd.concat(
        [pd.DataFrame([pair_row]), table, pd.DataFrame([blank_row])],
        ignore_index=True,
    )
    return block


def _append_blocks_excel(blocks: list[pd.DataFrame]) -> None:
    if not blocks:
        return

    os.makedirs(ESM_DIR, exist_ok=True)
    combined = pd.concat(blocks, ignore_index=True)

    if os.path.exists(ESM_OUTPUT):
        existing = pd.read_excel(ESM_OUTPUT, sheet_name="results")
        combined = pd.concat([existing, combined], ignore_index=True)

    with pd.ExcelWriter(ESM_OUTPUT, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="results", index=False)


def _extract_site_and_ssp(file_path: str) -> tuple[str, str | None]:
    base = os.path.splitext(os.path.basename(file_path))[0]
    tokens = base.replace("-", "_").split("_")
    ssp_token = None
    for token in tokens:
        if token.lower().startswith("ssp"):
            ssp_token = token.lower()
            break

    site = tokens[1] if len(tokens) > 1 else base
    return site, ssp_token


def _pair_display_name(file_a: str, file_b: str) -> str:
    site_a, ssp_a = _extract_site_and_ssp(file_a)
    site_b, ssp_b = _extract_site_and_ssp(file_b)
    site = site_a if site_a else site_b
    ssp_a = ssp_a or "ssp?"
    ssp_b = ssp_b or "ssp?"
    return f"{site} {ssp_a} vs {ssp_b}"


def _save_joint_pdf(a_vals: np.ndarray, b_vals: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    hexplot = ax.hexbin(
        a_vals,
        b_vals,
        gridsize=HIST_BINS,
        cmap="viridis",
        mincnt=1,
    )
    fig.colorbar(hexplot, ax=ax, label="count")
    ax.set_xlabel("file A (uas)")
    ax.set_ylabel("file B (uas)")
    ax.set_title(f"Joint PDF (Hexbin Histogram): {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0 or math.isnan(std):
        return np.full_like(x, float("nan"), dtype=float)
    coef = 1.0 / (std * math.sqrt(2.0 * math.pi))
    return coef * np.exp(-0.5 * ((x - mean) / std) ** 2)


def _save_normal_overlay(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    title: str,
    label_a: str,
    label_b: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    all_vals = np.concatenate([a_vals, b_vals])
    if all_vals.size == 0:
        return

    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    x = np.linspace(min_val, max_val, 300)

    ax.hist(a_vals, bins=HIST_BINS, density=True, alpha=0.35, label=f"{label_a} hist")
    ax.hist(b_vals, bins=HIST_BINS, density=True, alpha=0.35, label=f"{label_b} hist")

    mean_a = float(np.mean(a_vals)) if len(a_vals) else float("nan")
    std_a = float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else float("nan")
    mean_b = float(np.mean(b_vals)) if len(b_vals) else float("nan")
    std_b = float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else float("nan")

    ax.plot(x, _normal_pdf(x, mean_a, std_a), label=f"{label_a} normal")
    ax.plot(x, _normal_pdf(x, mean_b, std_b), label=f"{label_b} normal")

    ax.set_xlabel("uas")
    ax.set_ylabel("density")
    ax.set_title(f"Normal Distribution: {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _unique_path(folder: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    if not os.path.exists(candidate):
        return candidate

    index = 1
    while True:
        candidate = os.path.join(folder, f"{base}_{index:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def main() -> None:
    if not os.path.isdir(SSP_DIR):
        raise FileNotFoundError(f"Folder not found: {SSP_DIR}")

    os.makedirs(IMG_JOINT_DIR, exist_ok=True)
    os.makedirs(IMG_NORMAL_DIR, exist_ok=True)

    files = sorted(
        [
            os.path.join(SSP_DIR, name)
            for name in os.listdir(SSP_DIR)
            if name.lower().endswith(".xlsx")
        ]
    )

    if len(files) < 2:
        print("Need at least two .xlsx files in ssp_files to run analysis.")
        return

    pairs = list(itertools.combinations(files, 2))
    if not pairs:
        print("No file pairs found for analysis.")
        return

    excel_blocks: list[pd.DataFrame] = []

    for file_a, file_b in pairs:
        name_a = os.path.basename(file_a)
        name_b = os.path.basename(file_b)

        try:
            series_a = _load_series(file_a)
            series_b = _load_series(file_b)
        except ValueError as exc:
            print(f"Skipping pair {name_a} vs {name_b}: {exc}")
            continue

        stats_a = _series_stats(series_a)
        stats_b = _series_stats(series_b)
        pair = _pair_metrics(series_a, series_b)

        table = _build_table(name_a, name_b, stats_a, stats_b, pair)
        print(f"\nPair: {name_a} vs {name_b}")
        print(table.to_string(index=False))

        excel_blocks.append(_block_for_excel(name_a, name_b, table))

        n = min(len(series_a), len(series_b))
        if n >= MIN_ROWS:
            a_vals = series_a.iloc[:n].to_numpy()
            b_vals = series_b.iloc[:n].to_numpy()
            display_name = _pair_display_name(file_a, file_b)
            joint_path = _unique_path(IMG_JOINT_DIR, f"{display_name}.png")
            normal_path = _unique_path(IMG_NORMAL_DIR, f"{display_name}.png")
            _save_joint_pdf(a_vals, b_vals, display_name, joint_path)
            _save_normal_overlay(a_vals, b_vals, display_name, name_a, name_b, normal_path)

    _append_blocks_excel(excel_blocks)


if __name__ == "__main__":
    main()
