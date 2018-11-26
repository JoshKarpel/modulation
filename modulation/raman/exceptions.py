class RamanException(Exception):
    pass


class MissingVolumeIntegrator(RamanException):
    pass


class MissingRamanMaterial(RamanException):
    pass
