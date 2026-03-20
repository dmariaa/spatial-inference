# Results tables scripts

This bundle separates the workflow into two steps.

## 1) Compute all summary data from the raw experiment CSV

```bash
python compute_results_data.py --csv /path/to/results_with_stats.csv --outdir derived_tables_data
```

This writes summary CSV files such as:

- `overall_performance.csv`
- `friedman_results.csv`
- `wilcoxon_holm_results.csv`
- `group_mae_stats.csv`
- `group_mse_stats.csv`
- `temporal_hour.csv`
- `temporal_daytype.csv`
- `hardest_cases.csv`

## 2) Render LaTeX tables from those summary files

```bash
python render_results_tables.py --datadir derived_tables_data --outdir latex_tables
```

This writes `.tex` table files ready to `\input{...}` into the paper.

## Notes

- Group labels `G1..G10` are assigned by the first appearance order of `sensor_group` in the CSV, which matches the order used in the previous discussion.
- The temporal tables use the narrow vertical layout.
- The hardest-cases table uses the top 5% worst DIP cases for MAE and MSE.
