import os as _os
import numpy as _np
import pyximport as _pyximport

_pyx_dir = _os.path.join(_os.path.dirname(__file__), ".pyxbld")
_pyximport.install(
    setup_args={"include_dirs": _np.get_include()}, build_dir=_pyx_dir, language_level=3
)

from .sims import (
    RamanSpecification,
    RamanSimulation,
    StimulatedRamanScatteringSpecification,
    StimulatedRamanScatteringSimulation,
    RamanSidebandSpecification,
    RamanSidebandSimulation,
    FourWaveMixingSpecification,
    FourWaveMixingSimulation,
    AUTO_CUTOFF,
)
from .anim import SquareAnimator, PolarComplexAmplitudeAxis
from .pump import (
    MonochromaticPump,
    ConstantMonochromaticPump,
    RectangularMonochromaticPump,
)
from .evolve import (
    EvolutionAlgorithm,
    ForwardEuler,
    BackwardEuler,
    TrapezoidRule,
    RungeKutta4,
)
from .volume import ModeVolumeIntegrator
from .mode import Mode
from .material import RamanMaterial
from .lookback import Lookback
