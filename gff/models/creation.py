from pathlib import Path
import torch

import gff.normalisation


def get_norms(C):
    norms = {}
    data_folder = Path(C["data_folder"])

    era5_norm = gff.normalisation.get_era5_norm(data_folder, C["era5_keys"])
    era5_mean = torch.tensor(era5_norm["mean"].values).reshape((1, 1, -1, 1, 1))
    era5_std = torch.tensor(era5_norm["std"].values).reshape((1, 1, -1, 1, 1))
    norms["era5"] = (era5_mean, era5_std)

    era5l_norm = gff.normalisation.get_era5_land_norm(data_folder, C["era5_land_keys"])
    era5l_mean = torch.tensor(era5l_norm["mean"].values).reshape((1, 1, -1, 1, 1))
    era5l_std = torch.tensor(era5l_norm["std"].values).reshape((1, 1, -1, 1, 1))
    norms["era5_land"] = (era5l_mean, era5l_std)

    if "hydroatlas_basin" in C["data_sources"]:
        hydroatlas_norm = gff.normalisation.get_hydroatlas_norm(data_folder, C["hydroatlas_keys"])
        hydroatlas_mean = torch.tensor(hydroatlas_norm["mean"].values).reshape((1, -1, 1, 1))
        hydroatlas_std = torch.tensor(hydroatlas_norm["std"].values).reshape((1, -1, 1, 1))
        norms["hydroatlas_basin"] = (
            hydroatlas_mean,
            hydroatlas_std,
        )

    if "dem_local" in C["data_sources"] or "dem_context" in C["data_sources"]:
        dem_norm = gff.normalisation.get_dem_norm(data_folder, fold=C["fold"])
        dem_mean = torch.tensor(dem_norm["mean"].values).reshape((1, -1, 1, 1))
        dem_std = torch.tensor(dem_norm["std"].values).reshape((1, -1, 1, 1))
        norms["dem"] = (dem_mean, dem_std)

    if "hand" in C["data_sources"]:
        hand_norm = gff.normalisation.get_hand_norm(data_folder, fold=C["fold"])
        hand_mean = torch.tensor(hand_norm["mean"].values).reshape((1, -1, 1, 1))
        hand_std = torch.tensor(hand_norm["std"].values).reshape((1, -1, 1, 1))
        norms["hand"] = (hand_mean, hand_std)

    if "s1" in C["data_sources"]:
        s1_norm = gff.normalisation.get_s1_norm(data_folder, fold=C["fold"])
        s1_mean = torch.tensor(s1_norm["mean"].values).reshape((1, -1, 1, 1))
        s1_std = torch.tensor(s1_norm["std"].values).reshape((1, -1, 1, 1))
        norms["s1"] = (s1_mean, s1_std)

    return norms


def create(C):
    if C["model"] == "debug_model":
        from . import debug_model

        return debug_model.DebugModel()
    elif C["model"] == "two_utae":
        from . import two_utae

        norms = get_norms(C)

        return two_utae.TwoUTAE(
            C["era5_keys"],
            C["era5_land_keys"],
            C["hydroatlas_keys"],
            C["hydroatlas_dim"],
            C["lead_time_dim"],
            norms=norms,
            w_hydroatlas_basin=("hydroatlas_basin" in C["data_sources"]),
            w_dem_context=("dem_context" in C["data_sources"]),
            w_dem_local=("dem_local" in C["data_sources"]),
            w_s1=("s1" in C["data_sources"]),
            n_predict=C["n_classes"],
            weather_window_size=C["weather_window"],
            context_embed_output_dim=C["context_embed_output_dim"],
            temp_encoding=C["temp_encoding"],
        )
    else:
        raise NotImplementedError("Not a valid model name")
