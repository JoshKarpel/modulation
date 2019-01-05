import simulacra as si  # somewhere there's ordering in these imports
# import modulation     # will get nasty import error if not this first

from pathlib import Path

import pickle

import click

import htmap

from . import shared


@click.command(context_settings = shared.CLI_CONTEXT_SETTINGS)
@click.argument('map_id')
@click.option('--outdir', default = None)
def main(map_id, outdir):
    with shared.make_spinner('loading map...') as spinner:
        map = htmap.load(map_id)
        spinner.succeed('loaded map')

    if outdir is None:
        outdir = Path.cwd()

    with shared.make_spinner('loading sims...') as spinner:
        sims = tuple(map)
        spinner.succeed('loaded sims')

    with shared.make_spinner('pickling sims...') as spinner:
        with (outdir / f'{map_id}.sims').open(mode = 'wb') as f:
            pickle.dump(sims, f)
        spinner.succeed('pickled sims')


if __name__ == '__main__':
    main()
