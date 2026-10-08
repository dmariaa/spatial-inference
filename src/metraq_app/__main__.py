"""Click entry point for the local METRAQ web app."""
import os
import click
import uvicorn


@click.command()
@click.option('--host', default='127.0.0.1', show_default=True)
@click.option('--port', default=8501, type=click.IntRange(1, 65535), show_default=True)
@click.option('--source', type=click.Choice(['files', 'db']), default='files', show_default=True)
def main(host, port, source):
    """Open the METRAQ viewer using a Python API and native browser frontend."""
    os.environ['METRAQ_APP_SOURCE'] = source
    uvicorn.run('metraq_app.app:app', host=host, port=port)


if __name__ == '__main__':
    main()
