import simulacra as si
import simulacra.units as u

from .. import refraction, fmt

MATERIAL_DATA = {
    "silica": dict(
        modulation_omega=u.twopi * 14 * u.THz,
        raman_linewidth=1 * u.THz,
        coupling_prefactor=(
            0.75 * 1e-2 * ((u.atomic_electric_dipole_moment ** 2) / (500 * u.THz))
        ),
        number_density=5e22 / (u.cm ** 3),
        index_of_refraction=refraction.SellmeierIndex.from_name("silica"),
    )
}


class RamanMaterial:
    """A class that represents a Raman-active material."""

    def __init__(
        self,
        *,
        modulation_omega: float,
        coupling_prefactor: complex,
        raman_linewidth: float,
        number_density: float,
        index_of_refraction: refraction.IndexOfRefraction,
    ):
        self.modulation_omega = modulation_omega
        self.raman_prefactor = (coupling_prefactor ** 2) / (4 * (u.hbar ** 3))
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
            "Raman Linewidth", fmt.quantity(self.raman_linewidth, fmt.FREQUENCY_UNITS)
        )
        info.add_field("Number Density", self.number_density)
        info.add_info(self.index_of_refraction.info())

        return info
