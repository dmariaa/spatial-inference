from __future__ import annotations

from pathlib import Path

import click

from metraq_gnn.comparison import build_comparison, load_matched


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("--gnn-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--reference", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def main(gnn_root: Path, reference: Path, output_dir: Path) -> None:
    """Create an exact matched comparison against DIP, KRG and IDW."""
    combined = load_matched(gnn_root, reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison(combined)
    combined.to_csv(output_dir / "matched_predictions.csv", index=False)
    comparison["overall"].to_csv(output_dir / "metrics_overall.csv", index=False)
    comparison["by_group"].to_csv(output_dir / "metrics_by_group.csv", index=False)
    comparison["by_case"].to_csv(output_dir / "metrics_by_case.csv", index=False)
    comparison["paired"].to_csv(output_dir / "paired_mae_comparison.csv", index=False)
    click.echo(comparison["overall"].to_string(index=False))
    click.echo("\n" + comparison["paired"].to_string(index=False))


if __name__ == "__main__":
    main()
