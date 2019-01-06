import simulacra as si  # somewhere there's ordering in these imports
import modulation  # will get nasty import error if not this first

from pathlib import Path

import gzip
import pickle

import click

import htmap

from . import shared


@click.command(context_settings = shared.CLI_CONTEXT_SETTINGS)
@click.argument('map_id')
@click.option('--outdir', default = None)
def main(map_id, outdir):
    with shared.make_spinner(f'loading map {map_id}...') as spinner:
        map = htmap.load(map_id)
        spinner.succeed(f'loaded map {map_id}')

    if outdir is None:
        outdir = Path.cwd()
    outdir = Path(outdir)

    with shared.make_spinner(f'loading sims from {map_id}...') as spinner:
        sims = tuple(map)
        spinner.succeed(f'loaded sims from {map_id}')

    with shared.make_spinner(f'pickling sims from {map_id}...') as spinner:
        with gzip.open(outdir / f'{map_id}.sims', mode = 'wb') as f:
            pickle.dump(sims, f)
        spinner.succeed(f'pickled sims from {map_id}')


if __name__ == '__main__':
    main()
