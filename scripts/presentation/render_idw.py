"""Run with: uv run --group presentation python -m scripts.presentation.render_idw --help."""
from pathlib import Path

if __name__ == '__main__':
    import click

    @click.command()
    @click.option('--month', default='2024-01', show_default=True, help='Calendar month in YYYY-MM format.')
    @click.option('--output', type=click.Path(path_type=Path), default='outputs/idw_nine', show_default=True)
    @click.option('--data-cache', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help='Read existing reconstruction cache; reuse its map unless --map-cache is specified.')
    @click.option('--map-cache', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help='Optional existing map directory to import once, e.g. outputs/idw_month.')
    @click.option('--pollutant', 'pollutant_ids', multiple=True, type=int, help='Magnitude ID; repeat to choose 1-9 pollutants. Default: 8,7,12,9,10,14,6,30,20.')
    @click.option('--seconds', default=62, type=click.IntRange(1,180), show_default=True)
    @click.option('--fps', default=24, type=click.IntRange(1,30), show_default=True)
    @click.option('--config', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help='Visual YAML settings; defaults to scripts/presentation/style.yaml.')
    @click.option('--preview-only', is_flag=True, help='Generate start/middle/end previews and poster without FFmpeg.')
    def main(month, output, map_cache, data_cache, pollutant_ids, seconds, fps, config, preview_only):
        """Render synchronized IDW cubes with a moving 24-hour time window."""
        from scripts.presentation.data import DEFAULT_POLLUTANTS
        from scripts.presentation.rendering import render

        try:
            render(month, output, seconds, fps, preview_only, pollutant_ids or DEFAULT_POLLUTANTS, map_cache, config, data_cache)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc

    main()
