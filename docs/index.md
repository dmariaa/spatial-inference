# METRAQ Documentation

This site contains the internal experiment documentation for `spatial-inference`.

## Included Pages

- **Experiments Schema**: structure of session folders and file formats.
- **Experiment Results**: generated comparison tables across sessions under `output/experiments`.
- **Notes**: additional working notes.

## Generate Results Page

Regenerate the experiment results markdown before publishing:

```powershell
.\.venv\Scripts\python.exe src/metraq_dip/tools/generate_experiment_results_doc.py --experiments-root output/experiments --output-file docs/experiment_results.md
```

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
