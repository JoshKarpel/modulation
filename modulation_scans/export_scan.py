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
def main(tag, outdir):
    with shared.make_spinner(f'loading map {tag}...') as spinner:
        map = htmap.load(tag)
        spinner.succeed(f'loaded map {tag}')

    if outdir is None:
        outdir = Path.cwd()
    outdir = Path(outdir)

    with shared.make_spinner(f'loading sims from {tag}...') as spinner:
        sims = tuple(map)
        spinner.succeed(f'loaded sims from {tag}')

    with shared.make_spinner(f'pickling sims from {tag}...') as spinner:
        with gzip.open(outdir / f'{tag}.sims', mode = 'wb') as f:
            pickle.dump(sims, f)
        spinner.succeed(f'pickled sims from {tag}')


if __name__ == '__main__':
    main()
