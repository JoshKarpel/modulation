from typing import Set, Any, Iterable, List
import logging

from pathlib import Path
import gzip
import pickle

from tqdm import tqdm

import simulacra as si

logger = logging.getLogger(__name__)


class ParameterScan:
    def __init__(self, tag: str, sims: Iterable[si.Simulation]):
        self.tag = tag
        self.sims = list(sims)

    @classmethod
    def from_file(cls, path: Path):
        path = Path(path)

        sims = []
        with gzip.open(path, mode="rb") as f:
            for _ in tqdm(
                range(pickle.load(f))
            ):  # first entry is the number of entries
                sims.append(pickle.load(f))

        ps = cls(path.stem, sims)

        logger.debug(f"loaded {len(ps)} simulations from {path}")

        return ps

    def __str__(self):
        return f"{self.__class__.__name__}(tag = {self.tag})"

    def __len__(self):
        return len(self.sims)

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
