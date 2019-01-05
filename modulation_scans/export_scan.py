import simulacra as si  # somewhere there's ordering in these imports
# import modulation     # will get nasty import error if not this first

import pickle

import click

import htmap

from . import shared


@click.command(context_settings = shared.CLI_CONTEXT_SETTINGS)
@click.argument('map_id')
def main(map_id):
    with shared.make_spinner('loading map...') as spinner:
        map = htmap.load(map_id)
        spinner.succeed('loaded map')

    with shared.make_spinner('pickling sims...') as spinner:
        pickle.dump(tuple(map), f'{map_id}.sims')
        spinner.succeed('pickled sims')


if __name__ == '__main__':
    main()
