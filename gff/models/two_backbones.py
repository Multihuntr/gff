import torch
import torch.nn as nn
import torch.nn.functional as F

import gff.models.utae as utae
import gff.models.metnet as metnet
import gff.models.unet3d as unet3d
import gff.models.logistic_regression as lr_model


def get_empty_norms(era5_bands: int, era5l_bands: int, glofas_bands: int, hydroatlas_bands: int):
    return {
        "era5": (torch.zeros((1, 1, era5_bands, 1, 1)), torch.ones((1, 1, era5_bands, 1, 1))),
        "era5_land": (
            torch.zeros((1, 1, era5l_bands, 1, 1)),
            torch.ones((1, 1, era5l_bands, 1, 1)),
        ),
        "glofas": (
            torch.zeros((1, 1, glofas_bands, 1, 1)),
            torch.ones((1, 1, glofas_bands, 1, 1)),
        ),
        "hydroatlas_basin": (
            torch.zeros((1, hydroatlas_bands, 1, 1)),
            torch.ones((1, hydroatlas_bands, 1, 1)),
        ),
        "dem": (torch.zeros((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),
        "hand": (torch.zeros((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),
        "s1": (torch.zeros((1, 2, 1, 1)), torch.ones((1, 2, 1, 1))),
    }


options_metnet = {
    "dim": 64,
    "attn_depth": 12,
    "attn_dim_head": 64,
    "attn_heads": 16,
    "attn_dropout": 0.1,
    "vit_window_size": 8,
    "vit_mbconv_expansion_rate": 4,
    "vit_mbconv_shrinkage_rate": 0.25,
    "resnet_block_depth": 2,
}
# "input_2496_channels" : context_embed_input_dim,
# input_4996_channels : 16 + 1,
# "surface_and_hrrr_target_spatial_size" : 128,
# "hrrr_channels" : 256,
# crop_size_post_16km : 48,


class ModelBackbones(nn.Module):
    def __init__(
        self,
        era5_bands,
        era5l_bands,
        glofas_bands=[],
        hydroatlas_bands=[],
        hydroatlas_dim=None,
        lead_time_dim=None,
        norms={},
        w_era5=True,
        w_era5_land=True,
        w_glofas=False,
        w_hydroatlas_basin=True,
        w_dem_context=True,
        w_dem_local=False,
        w_hand=True,
        w_s1=True,
        derive_land_sea_from_dem=False,
        n_predict=3,
        weather_window_size=20,
        context_embed_output_dim=5,
        center_crop_context=True,
        average_context=True,
        backbone="utae",
        nan_impute_val="zero",
        cond_norm_affine=None,
    ):
        super().__init__()
        self.era5_bands = era5_bands
        self.era5l_bands = era5l_bands
        self.glofas_bands = glofas_bands
        self.hydroatlas_bands = hydroatlas_bands
        self.w_era5 = w_era5
        self.w_era5_land = w_era5_land
        self.w_glofas = w_glofas
        self.w_hydroatlas_basin = w_hydroatlas_basin
        self.w_dem_context = w_dem_context
        self.w_dem_local = w_dem_local
        self.w_hand = w_hand
        self.w_s1 = w_s1
        self.derive_land_sea_from_dem = derive_land_sea_from_dem
        self.n_predict = n_predict
        self.weather_window_size = weather_window_size
        self.center_crop_context = center_crop_context
        self.average_context = average_context
        self.backbone = backbone
        self.nan_impute_val = nan_impute_val

        # Store normalisation info on model
        # (To load model weights, the shapes must be identical; so use empty if not known at init)
        empty_norms = get_empty_norms(
            len(era5_bands), len(era5l_bands), len(glofas_bands), len(hydroatlas_bands)
        )
        for key in ["era5", "era5_land", "hydroatlas_basin", "glofas", "dem", "s1", "hand"]:
            if key in norms:
                mean, std = norms[key]
            else:
                mean, std = empty_norms[key]
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std", std)

        assert (
            self.w_s1 or self.w_dem_local or self.w_hand
        ), "Must provide one of s1, dem local or hand to produce local scale predictions"
        assert (self.w_s1 and (lead_time_dim is not None)) or (
            (not self.w_s1) and (lead_time_dim is None)
        ), "If you provide s1, you must also provide lead_time_dim. If not, you shouldn't."

        # Determine context embedding sizes
        self.context_embed_input_dim = 0
        if self.w_era5:
            self.context_embed_input_dim += len(era5_bands)
        if self.w_era5_land:
            self.context_embed_input_dim += len(era5l_bands)
        if self.w_glofas:
            self.context_embed_input_dim += len(glofas_bands)

        if self.w_hydroatlas_basin:
            self.n_hydroatlas = len(hydroatlas_bands)
            self.hydro_atlas_embed = nn.Conv2d(
                self.n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1
            )
            self.context_embed_input_dim += hydroatlas_dim
        if self.w_dem_context:
            self.context_embed_input_dim += 1
            if self.derive_land_sea_from_dem:
                self.context_embed_input_dim += 1

        # Determine local sizes
        if self.context_embed_input_dim > 0:
            self.local_input_dim = context_embed_output_dim
        else:
            self.local_input_dim = 0
        if self.w_dem_local:
            self.local_input_dim += 1
            if self.derive_land_sea_from_dem:
                self.local_input_dim += 1
        if self.w_hand:
            self.local_input_dim += 1
        if self.w_s1:
            self.local_input_dim += 2
            self.len_lead = weather_window_size + 1
            self.lead_time_embedding = nn.Embedding(self.len_lead, lead_time_dim)

        # Instantiate backbones
        if backbone == "utae":
            if self.context_embed_input_dim > 0:
                self.context_embed = utae.UTAE(
                    self.context_embed_input_dim,
                    encoder_widths=[32, 32],
                    decoder_widths=[32, 32],
                    out_conv=[context_embed_output_dim],
                    cond_dim=lead_time_dim,
                    temp_encoding="ltae",
                    cond_norm_affine=cond_norm_affine,
                )
            self.local_embed = utae.UTAE(
                self.local_input_dim,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[64, 64, 64, 128],
                out_conv=[64, n_predict],
                cond_dim=lead_time_dim,
                temp_encoding="ltae",
                cond_norm_affine=cond_norm_affine,
            )
        elif backbone == "recunet_lstm":
            if self.context_embed_input_dim > 0:
                self.context_embed = utae.UTAE(
                    self.context_embed_input_dim,
                    encoder_widths=[32, 32],
                    decoder_widths=[32, 32],
                    out_conv=[context_embed_output_dim],
                    cond_dim=lead_time_dim,
                    temp_encoding="lstm",
                    cond_norm_affine=cond_norm_affine,
                )
            self.local_embed = utae.UTAE(
                self.local_input_dim,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[64, 64, 64, 128],
                out_conv=[64, n_predict],
                cond_dim=lead_time_dim,
                temp_encoding="lstm",
                cond_norm_affine=cond_norm_affine,
            )
        elif backbone == "metnet":
            options_metnet["lead_time_embed_dim"] = lead_time_dim
            options_metnet["cond_norm_affine"] = cond_norm_affine

            options_metnet["dim_in"] = weather_window_size * self.context_embed_input_dim
            options_metnet["out_conv"] = context_embed_output_dim
            if self.context_embed_input_dim > 0:
                self.context_embed = metnet.MetNet3(**options_metnet)

            options_metnet["dim_in"] = 1 * self.local_input_dim
            options_metnet["out_conv"] = n_predict
            self.local_embed = metnet.MetNet3(**options_metnet)
        elif backbone == "3dunet":
            if self.context_embed_input_dim > 0:
                self.context_embed = unet3d.UNet3D(
                    input_dim=self.context_embed_input_dim,
                    out_conv=context_embed_output_dim,
                    cond_dim=lead_time_dim,
                    op_type="3d",
                    cond_norm_affine=cond_norm_affine,
                )
            self.local_embed = unet3d.UNet3D(
                input_dim=self.local_input_dim,
                out_conv=n_predict,
                cond_dim=lead_time_dim,
                op_type="2d",
                cond_norm_affine=cond_norm_affine,
            )
        elif backbone == "LR":
            assert self.context_embed_input_dim == 0, "Logistic regression is local-only"
            self.local_embed = lr_model.LogisticRegression(
                n_channels=self.local_input_dim, out_channels=n_predict
            )
        elif backbone == "UTAE+LR":
            self.context_embed = utae.UTAE(
                self.context_embed_input_dim,
                encoder_widths=[32, 32],
                decoder_widths=[32, 32],
                out_conv=[context_embed_output_dim],
                cond_dim=lead_time_dim,
                temp_encoding="ltae",
                cond_norm_affine=cond_norm_affine,
            )
            self.local_embed = lr_model.LogisticRegression(
                n_channels=self.local_input_dim, out_channels=n_predict
            )
        else:
            raise NotImplementedError(f"Unknown model name: {backbone}")

    def normalise(self, ex, key, suffix=None):
        if suffix is not None:
            ex_key = f"{key}_{suffix}"
        else:
            ex_key = key
        if ex_key in ex:
            data = ex[ex_key]
            mean = getattr(self, f"{key}_mean")
            std = getattr(self, f"{key}_std")
            data = (data - mean) / std
            return data

    def impute_nans(self, t: torch.Tensor | None):
        if t is not None:
            nan_mask = torch.isnan(t)
            t[nan_mask] = 0
            if self.nan_impute_val == "zero":
                pass  # already done
            elif self.nan_impute_val == "instance_mean":
                # Different mean per batch instance and channel,
                # but channel is in a different position for different data shapes
                if len(t.shape) == 5:
                    dim = (1, 3, 4)
                else:
                    dim = (2, 3)
                val = torch.nanmean(t, dim=dim, keepdim=True)
                t += nan_mask * val
            return t

    def get_lead_time_idx(self, lead):
        # Get the index into the embedding based on lead.
        # For lead times within the weather window size, lead == idx
        lead_copy = lead.clone()
        # For leads too far away in time we just set it to the last idx
        # This allows the model to know how to use soil moisture in the S1 images,
        # regardless of how old. e.g. the model can ignore it if it wants.
        lead_copy[lead > self.len_lead] = self.len_lead - 1
        return lead_copy

    def derive_landsea(self, dem):
        if dem is not None:
            return ~torch.isnan(dem)
        else:
            return None

    def get_context_shape(self, ex):
        dynamics_keys = ["era5", "era5_land", "glofas"]
        statics_keys = ["hydroatlas_basin", "dem_context"]
        for dk in dynamics_keys:
            if dk in ex:
                B, N, _, cH, cW = ex[dk].shape
                return B, N, cH, cW
        for sk in statics_keys:
            if sk in ex:
                B, _, cH, cW = ex[sk].shape
                N = 1
                return B, N, cH, cW

    def forward(self, ex):
        for k in ["s1", "dem_local", "hand"]:
            if k in ex:
                example_local = ex[k]
                break
        B, _, fH, fW = example_local.shape

        # Normalise inputs
        era5_inp = self.normalise(ex, "era5")
        era5l_inp = self.normalise(ex, "era5_land")
        glofas_inp = self.normalise(ex, "glofas")
        hydroatlas_inp = self.normalise(ex, "hydroatlas_basin")
        dem_context_inp = self.normalise(ex, "dem", "context")
        derived_landsea_context = self.derive_landsea(dem_context_inp)
        dem_local_inp = self.normalise(ex, "dem", "local")
        derived_landsea_local = self.derive_landsea(dem_local_inp)
        hand_inp = self.normalise(ex, "hand")
        s1_inp = self.normalise(ex, "s1")

        # These inputs might have nan
        era5l_inp = self.impute_nans(era5l_inp)
        glofas_inp = self.impute_nans(glofas_inp)
        hydroatlas_inp = self.impute_nans(hydroatlas_inp)
        dem_context_inp = self.impute_nans(dem_context_inp)
        dem_local_inp = self.impute_nans(dem_local_inp)
        hand_inp = self.impute_nans(hand_inp)

        # Get Lead time embeddings
        if self.w_s1:
            lead_idx = self.get_lead_time_idx(ex["s1_lead_days"])
            lead = self.lead_time_embedding(lead_idx)
        else:
            lead = None

        # Process context inputs
        if self.context_embed_input_dim > 0:
            B, N, cH, cW = self.get_context_shape(ex)
            batch_positions = (
                torch.arange(0, N).reshape((1, N)).repeat((B, 1)).to(example_local.device)
            )
            context_statics_lst = []
            if self.w_hydroatlas_basin:
                embedded_hydro_atlas_raster = self.hydro_atlas_embed(hydroatlas_inp)
                context_statics_lst.append(embedded_hydro_atlas_raster)
            if self.w_dem_context:
                context_statics_lst.append(dem_context_inp)
                if self.derive_land_sea_from_dem:
                    context_statics_lst.append(derived_landsea_context)
            # Add the statics in at every step
            context_lst = []
            if self.w_era5_land:
                context_lst.append(era5l_inp)
            if self.w_era5:
                context_lst.append(era5_inp)
            if self.w_glofas:
                context_lst.append(glofas_inp)
            if self.w_hydroatlas_basin or self.w_dem_context:
                context_statics = (
                    torch.cat(context_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
                )
                context_lst.append(context_statics)
            context_inp = torch.cat(context_lst, dim=2)
            context_embedded = self.context_embed(
                context_inp, batch_positions=batch_positions, lead=lead
            )
            if self.backbone != "3dunet":
                context_embedded = context_embedded[:, 0]

            # Select the central 2x2 pixels and average
            if self.center_crop_context:
                ylo, yhi = cH // 2 - 1, cH // 2 + 3
                xlo, xhi = cW // 2 - 1, cW // 2 + 3
            else:
                ylo, yhi = 0, cH
                xlo, xhi = 0, cW
            context_out = context_embedded[:, :, ylo:yhi, xlo:xhi]
            if self.average_context:
                context_out = context_out.mean(axis=(2, 3), keepdims=True)
            context_out_upsc = F.interpolate(context_out, size=(fH, fW), mode="nearest")

            # Context prepared for
            local_lst = [context_out_upsc]
        else:
            batch_positions = (
                torch.arange(0, 1).reshape((1, 1)).repeat((B, 1)).to(example_local.device)
            )
            local_lst = []

        # Process local inputs
        if self.w_s1:
            local_lst.append(s1_inp)
        if self.w_dem_local:
            local_lst.append(dem_local_inp)
            if self.derive_land_sea_from_dem:
                local_lst.append(derived_landsea_local)
        if self.w_hand:
            local_lst.append(hand_inp)
        local_inp = torch.cat(local_lst, dim=1)
        # Pretend it's temporal data with one time step
        local_inp = local_inp[:, None]
        out = self.local_embed(local_inp, batch_positions=batch_positions[:, :1], lead=lead)

        if self.backbone != "3dunet":
            out = out[:, 0]
        return out


if __name__ == "__main__":
    # For debugging
    B, T = 8, 12
    cH, cW = 32, 32
    fH, fW = 224, 224
    n_hydroatlas = 128
    hydroatlas_dim = 16
    n_era5 = 16
    n_era5_land = 8
    n_glofas = 3
    lead_time_dim = 16
    ex = {
        "era5": torch.randn((B, T, n_era5, cH, cW)).cuda(),
        "era5_land": torch.randn((B, T, n_era5_land, cH, cW)).cuda(),
        "glofas": torch.randn((B, T, n_glofas, cH, cW)).cuda(),
        "hydroatlas_basin": torch.randn((B, n_hydroatlas, cH, cW)).cuda(),
        "dem_context": torch.randn((B, 1, cH, cW)).cuda(),
        "s1": torch.randn(B, 2, fH, fW).cuda(),
        "dem_local": torch.randn((B, 1, fH, fW)).cuda(),
        "hand": torch.randn((B, 1, fH, fW)).cuda(),
        "s1_lead_days": torch.randint(0, 20, (B,)).cuda(),
    }
    era5_bands = list(range(n_era5))
    era5l_bands = list(range(n_era5_land))
    glofas_bands = list(range(n_glofas))
    hydroatlas_bands = list(range(n_hydroatlas))
    to_remove = [
        {},
        {"w_era5": False},
        {"w_era5_land": False},
        {"w_glofas": True},
        {"w_hydroatlas_basin": False},
        {"w_dem_context": False},
        {"w_s1": False},
        {"w_dem_local": False},
        {"w_hand": False},
        {
            "w_era5": False,
            "w_era5_land": False,
            "w_glofas": False,
            "w_hydroatlas_basin": False,
            "w_dem_context": False,
        },
    ]

    for d in to_remove:
        for backbone in ["utae", "recunet_lstm", "metnet", "3dunet"]:
            if d.get("w_s1", True):
                lead = {"lead_time_dim": lead_time_dim}
            else:
                lead = {}
            model = ModelBackbones(
                era5_bands,
                era5l_bands,
                glofas_bands,
                hydroatlas_bands,
                hydroatlas_dim,
                **lead,
                **d,
                weather_window_size=T,
                backbone=backbone,
            ).cuda()
            out = model(ex)
            assert out.shape == (8, 3, 224, 224)
            print("🗸", end="")

    lead = {"lead_time_dim": lead_time_dim}
    model = ModelBackbones(
        era5_bands,
        era5l_bands,
        glofas_bands,
        hydroatlas_bands,
        hydroatlas_dim,
        **lead,
        **to_remove[-1],
        weather_window_size=T,
        backbone="LR",
    ).cuda()
    out = model(ex)
    assert out.shape == (8, 3, 224, 224)
    print("🗸", end="")
    print()
    lr_model.print_weights(model.local_embed, ["S1 (0)", "S2 (0)", "HAND"])
