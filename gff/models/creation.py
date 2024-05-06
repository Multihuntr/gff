def create(C):
    n_hydroatlas = len(C["hydroatlas_keys"])
    n_era5 = len(C["era5_keys"])
    n_era5_land = len(C["era5_land_keys"])
    if C["model"] == "debug_model":
        from . import debug_model

        return debug_model.DebugModel()
    elif C["model"] == "two_utae":
        from . import two_utae

        return two_utae.TwoUTAE(
            (n_era5 + n_era5_land),
            n_hydroatlas,
            C["hydroatlas_dim"],
            C["lead_time_dim"],
            w_hydroatlas_basin=("hydroatlas_basin" in C["data_sources"]),
            w_dem_context=("dem_context" in C["data_sources"]),
            w_dem_local=("dem_local" in C["data_sources"]),
            w_s1=("s1" in C["data_sources"]),
        )
    else:
        raise NotImplementedError("Not a valid model name")
