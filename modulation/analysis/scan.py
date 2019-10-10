from typing import Set, Any, Iterable, List, Dict
import logging

from pathlib import Path
import gzip
import pickle

from tqdm import tqdm

import simulacra as si

logger = logging.getLogger(__name__)

CACHE: "Dict[Path, ParameterScan]" = {}


class ParameterScan:
    def __init__(self, tag: str, sims: Iterable[si.Simulation]):
        self.tag = tag
        self.sims = list(sims)

    @classmethod
    def from_file(cls, path: Path, show_progress=False):
        path = Path(path).absolute()

        if path in CACHE:
            return CACHE[path]

        with gzip.open(path, mode="rb") as f:
            first = pickle.load(f)

            # first entry is the number of entries
            if isinstance(first, int):
                sims = []
                it = range(first)
                if show_progress:
                    it = tqdm(it)
                for _ in it:
                    sims.append(pickle.load(f))
            else:  # it's just a list of sims
                sims = first

        ps = cls(path.stem, sims)

        logger.debug(f"loaded {len(ps)} simulations from {path}")

        CACHE[path] = ps

        return ps

    def __str__(self):
        return f"{self.__class__.__name__}(tag = {self.tag})"

    def __len__(self):
        return len(self.sims)

    def __iter__(self):
        yield from self.sims

    def __getitem__(self, item):
        return self.sims[item]

    def parameter_set(self, parameter: str) -> Set[Any]:
        return {getattr(sim.spec, parameter) for sim in self.sims}

    def select(self, **parameters) -> List[si.Simulation]:
        return sorted(
            (
                sim
                for sim in self.sims
                if all(getattr(sim.spec, k) == v for k, v in parameters.items())
            ),
            key=lambda sim: tuple(getattr(sim.spec, k) for k in parameters.keys()),
        )
