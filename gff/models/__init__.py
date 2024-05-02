from . import debug_model
from . import two_utae


def create(C):
    n_hydroatlas = len(C["hydroatlas_keys"])
    n_era5 = len(C["era5_keys"])
    n_era5_land = len(C["era5_land_keys"])
    if C["model"] == "debug_model":
        return debug_model.DebugModel()
    elif C["model"] == "two_utae":
        return two_utae.TwoUTAE(n_hydroatlas, C["hydroatlas_dim"], (n_era5 + n_era5_land))
    else:
        raise NotImplementedError("Not a valid model name")
