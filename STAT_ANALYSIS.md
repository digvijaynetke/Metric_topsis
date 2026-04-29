# Statistical Analysis (ssp_files)

## What it does

- Finds every .xlsx file inside the ssp_files folder.
- Builds every pair of files (all combinations).
- Reads the E column labeled "uas" from the first sheet.
- Prints one table per pair in the terminal.
- Appends each run's results into ESM/ssp_stat_results.xlsx (no overwrite).
- Saves charts for each pair into image folders without overwriting existing images.

## Metrics

Per file:
- mean
- mode
- median
- standard deviation
- kurtosis
- skewness

Per pair:
- correlation coefficient (Pearson r)
- RMSE
- mean error (mean of A - B)

## How to run

```bash
python ssp_stat_analysis.py
```

## Workflow

1) Put your .xlsx files in ssp_files.
2) Make sure each file has a column labeled "uas" (E column).
3) Run the script.
4) Read the tables for each file pair in the terminal.
5) Check ESM/ssp_stat_results.xlsx for appended output blocks.
6) Check image folders for charts.

## Notes

- If files have different lengths, the script aligns by the shortest length.
- If a file is missing the "uas" column, that pair is skipped.
- Correlation is Pearson r.
- Joint PDF images are saved in img_joint_pdf.
- Normal distribution overlays are saved in img_normal_dist.
- If an image name already exists, a numeric suffix is added.
