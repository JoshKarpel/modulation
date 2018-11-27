import simulacra.units as u

LENGTH_UNITS = (
    (u.nm, 'nm'),
    (u.um, 'um'),
)
INVERSE_LENGTH_UNITS = (
    (u.per_nm, '1/nm'),
)
TIME_UNITS = (
    (u.nsec, 'ns'),
    (u.usec, 'us'),
)
FREQUENCY_UNITS = (
    (u.THz, 'THz'),
    (u.GHz, 'GHz'),
    (u.MHz, 'MHz'),
)
ELECTRIC_FIELD_UNITS = (
    (u.V_per_m, 'V/m'),
)
ENERGY_UNITS = (
    (u.pJ, 'pJ'),
)
ANGLE_UNITS = (
    (u.pi, '𝜋 rad'),
    (u.deg, 'deg'),
)
CHARGE_UNITS = (
    (u.proton_charge, 'e'),
)
VOLUME_UNITS = (
    (u.um ** 3, 'um^3'),
)
POWER_UNITS = (
    (u.uW, 'µW'),
    (u.mW, 'mW'),
    (u.W, 'W'),
)


def quantity(quantity, units: tuple):
    """Format a single quantity for multiple units, as in :class:`simulacra.Info` fields."""
    return ' | '.join(f'{quantity / v:.4g} {s}' for v, s in units)
