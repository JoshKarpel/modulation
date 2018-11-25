class PowerScanResult:
    def __init__(self, sim):
        self.energies = sim.mode_energies(sim.mode_amplitudes)
        self.output_powers = sim.mode_output_powers(sim.mode_amplitudes)

        self.pump_power = sim.spec._pump_power


class PowerScanData:
    def __init__(self, sims):
        self.results = [PowerScanResult(sim) for sim in sims]

        d = sims[0]
        self.modes = d.spec.modes
        self.wavelength_bounds = d.spec._wavelength_bounds
        self.pump_mode = d.spec._pump_mode
        self.pump_wavelength = d.spec._pump_wavelength
