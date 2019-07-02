import simulacra as si
import simulacra.units as u

from .. import refraction, fmt

SILICA_MODULATION_OMEGA = u.twopi * 14 * u.THz
SILICA_RAMAN_LINEWIDTH = u.twopi * 2 * u.THz

MATERIAL_DATA = {
    "silica": dict(
        modulation_omega=SILICA_MODULATION_OMEGA,
        raman_linewidth=SILICA_RAMAN_LINEWIDTH,
        coupling_prefactor_squared=2
        * (u.c ** 2)
        * (u.epsilon_0 ** 2)
        * (u.hbar ** 3)
        * (1.4496 ** 2)
        * SILICA_RAMAN_LINEWIDTH
        * (1e-11 * u.cm / u.W)
        / (5e22 / (u.cm ** 3) * (u.twopi * u.c / (1064 * u.nm))),
        number_density=5e22 / (u.cm ** 3),
        index_of_refraction=refraction.SellmeierIndex.from_name("silica"),
    ),
    "silica-narrow": dict(
        modulation_omega=SILICA_MODULATION_OMEGA,
        raman_linewidth=SILICA_RAMAN_LINEWIDTH / 1e3,
        coupling_prefactor_squared=2
        * (u.c ** 2)
        * (u.epsilon_0 ** 2)
        * (u.hbar ** 3)
        * (1.4496 ** 2)
        * (SILICA_RAMAN_LINEWIDTH / 1e3)
        * (1e-11 * u.cm / u.W)
        / (5e22 / (u.cm ** 3) * (u.twopi * u.c / (1064 * u.nm))),
        number_density=5e22 / (u.cm ** 3),
        index_of_refraction=refraction.SellmeierIndex.from_name("silica"),
    ),
    "silica-very-narrow": dict(
        modulation_omega=SILICA_MODULATION_OMEGA,
        raman_linewidth=SILICA_RAMAN_LINEWIDTH / 1e6,
        coupling_prefactor_squared=2
        * (u.c ** 2)
        * (u.epsilon_0 ** 2)
        * (u.hbar ** 3)
        * (1.4496 ** 2)
        * (SILICA_RAMAN_LINEWIDTH / 1e6)
        * (1e-11 * u.cm / u.W)
        / (5e22 / (u.cm ** 3) * (u.twopi * u.c / (1064 * u.nm))),
        number_density=5e22 / (u.cm ** 3),
        index_of_refraction=refraction.SellmeierIndex.from_name("silica"),
    ),
}


class RamanMaterial:
    """A class that represents a Raman-active material."""

    def __init__(
        self,
        *,
        modulation_omega: float,
        coupling_prefactor_squared: complex,
        raman_linewidth: float,
        number_density: float,
        index_of_refraction: refraction.IndexOfRefraction,
    ):
        self.coupling_prefactor_squared = coupling_prefactor_squared

        self.modulation_omega = modulation_omega
        self.raman_prefactor = coupling_prefactor_squared / (4 * (u.hbar ** 3))
        self.raman_linewidth = raman_linewidth
        self.number_density = number_density
        self.index_of_refraction = index_of_refraction

    @classmethod
    def from_name(cls, name: str) -> "RamanMaterial":
        return cls(**MATERIAL_DATA[name])

    @property
    def modulation_frequency(self):
        return self.modulation_omega / u.twopi

    def info(self) -> si.Info:
        info = si.Info(header="Raman Material Properties")

        info.add_field("Raman Coupling Prefactor", self.raman_prefactor)
        info.add_field(
            "Raman Modulation Frequency",
            fmt.quantity(self.modulation_frequency, fmt.FREQUENCY_UNITS),
        )
        info.add_field(
            "Raman Linewidth",
            fmt.quantity(self.raman_linewidth / u.twopi, fmt.FREQUENCY_UNITS),
        )
        info.add_field("Number Density", self.number_density)
        info.add_info(self.index_of_refraction.info())

        return info
