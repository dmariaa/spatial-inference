# METRAQ Documentation

This site contains the internal experiment documentation for `spatial-inference`.

## Included Pages

- **Experiments**: entry point for experiment result blocks and artifact schema.
- **METRAQ NO**: paper-ready NO summary tables, window diagnostics, and exemplary window visualizations.
- **Cross-Dataset UNet**: UNet NO2/NOX comparisons for AIRPARIF and METRAQ.
- **Cross-Dataset Autoencoder**: autoencoder NO2/NOX comparisons for AIRPARIF and METRAQ.

## Build The Docs Site

Install docs dependencies:

```powershell
uv sync --group docs
```

Build the static site:

```powershell
uv run mkdocs build
```

Output is generated in `site/`.

## Local Preview

Run the local docs server:

```powershell
uv run mkdocs serve
```
