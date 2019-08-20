import os as _os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
_os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from .version import __version__, version, version_info


from . import raman, refraction, analysis, resonators
