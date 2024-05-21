import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import utae
import gff.models.utae as utae
import gff.models.unet3d as unet3d


def nans_to_zero(t: torch.Tensor | None):
    if t is not None:
        t[torch.isnan(t)] = 0
        return t


def get_empty_norms(era5_bands: int, era5l_bands: int, hydroatlas_bands: int):
    return {
        "era5": (torch.zeros((1, 1, era5_bands, 1, 1)), torch.ones((1, 1, era5_bands, 1, 1))),
        "era5_land": (
            torch.zeros((1, 1, era5l_bands, 1, 1)),
            torch.ones((1, 1, era5l_bands, 1, 1)),
        ),
        "hydroatlas_basin": (
            torch.zeros((1, hydroatlas_bands, 1, 1)),
            torch.ones((1, hydroatlas_bands, 1, 1)),
        ),
        "dem": (torch.zeros((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),
        "hand": (torch.zeros((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),
        "s1": (torch.zeros((1, 2, 1, 1)), torch.ones((1, 2, 1, 1))),
    }


class TwoUTAE(nn.Module):
    def __init__(
        self,
        era5_bands,
        era5l_bands,
        hydroatlas_bands=[],
        hydroatlas_dim=None,
        lead_time_dim=None,
        norms={},
        w_hydroatlas_basin=True,
        w_dem_context=True,
        w_dem_local=False,
        w_hand=True,
        w_s1=True,
        n_predict=3,
        weather_window_size=20,
        context_embed_output_dim=5,
        center_crop_context=True,
        average_context=True,
        temp_encoding="ltae",
    ):
        super().__init__()
        self.era5_bands = era5_bands
        self.era5l_bands = era5l_bands
        self.hydroatlas_bands = hydroatlas_bands
        self.w_hydroatlas_basin = w_hydroatlas_basin
        self.w_dem_context = w_dem_context
        self.w_dem_local = w_dem_local
        self.w_hand = w_hand
        self.w_s1 = w_s1
        self.n_predict = n_predict
        self.weather_window_size = weather_window_size
        self.center_crop_context = center_crop_context
        self.average_context = average_context
        self.temp_encoding = temp_encoding

        # Store normalisation info on model
        # (To load model weights, the shapes must be identical; so use empty if not known at init)
        empty_norms = get_empty_norms(len(era5_bands), len(era5l_bands), len(hydroatlas_bands))
        for key in ["era5", "era5_land", "hydroatlas_basin", "dem", "s1", "hand"]:
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

        # Create context embedding layers
        self.n_weather = len(era5_bands) + len(era5l_bands)
        self.n_hydroatlas = len(hydroatlas_bands)
        context_embed_input_dim = self.n_weather
        if self.w_hydroatlas_basin:
            self.hydro_atlas_embed = nn.Conv2d(
                self.n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1
            )
            context_embed_input_dim += hydroatlas_dim
        if self.w_dem_context:
            context_embed_input_dim += 1
        self.context_embed = utae.UTAE(
            context_embed_input_dim,
            encoder_widths=[32, 32],
            decoder_widths=[32, 32],
            out_conv=[context_embed_output_dim],
            temp_encoding=self.temp_encoding,
        )

        # Create local embedding/prediction layers
        local_input_dim = context_embed_output_dim
        if self.w_dem_local:
            local_input_dim += 1
        if self.w_hand:
            local_input_dim += 1
        if self.w_s1:
            local_input_dim += 2
            self.len_lead = weather_window_size + 1
            self.lead_time_embedding = nn.Embedding(self.len_lead, lead_time_dim)
        self.local_embed = utae.UTAE(
            local_input_dim,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[64, 64, 64, 128],
            out_conv=[64, n_predict],
            cond_dim=lead_time_dim,
            temp_encoding=self.temp_encoding,
        )

    def normalise(self, ex, key, suffix=None):
        if suffix is not None:
            ex_key = f"{key}_{suffix}"
        else:
            ex_key = key
        if ex_key in ex:
            data = ex[ex_key]
            km = f"{key}_mean"
            ks = f"{key}_std"
            data = (data - getattr(self, km)) / getattr(self, ks)
            return data

    def get_lead_time_idx(self, lead):
        # Get the index into the embedding based on lead.
        # For lead times within the weather window size, lead == idx
        lead_copy = lead.clone()
        # For leads too far away in time we just set it to the last idx
        # This allows the model to know how to use soil moisture in the S1 images,
        # regardless of how old. e.g. the model can ignore it if it wants.
        lead_copy[lead > self.len_lead] = self.len_lead - 1
        return lead_copy

    def forward(self, ex):
        B, N, cC, cH, cW = ex["era5"].shape
        batch_positions = torch.arange(0, N).reshape((1, N)).repeat((B, 1)).to(ex["era5"].device)
        if self.w_s1:
            example_local = ex["s1"]
        else:
            example_local = ex["dem_local"]
        fH, fW = example_local.shape[-2:]

        # Normalise inputs
        era5_inp = self.normalise(ex, "era5")
        era5l_inp = self.normalise(ex, "era5_land")
        hydroatlas_inp = self.normalise(ex, "hydroatlas_basin")
        dem_context_inp = self.normalise(ex, "dem", "context")
        dem_local_inp = self.normalise(ex, "dem", "local")
        hand_inp = self.normalise(ex, "hand")
        s1_inp = self.normalise(ex, "s1")

        # These inputs might have nan
        era5l_inp = nans_to_zero(era5l_inp)
        hydroatlas_inp = nans_to_zero(hydroatlas_inp)
        dem_context_inp = nans_to_zero(dem_context_inp)
        dem_local_inp = nans_to_zero(dem_local_inp)
        hand_inp = nans_to_zero(hand_inp)

        # Process context inputs
        context_statics_lst = []
        if self.w_hydroatlas_basin:
            embedded_hydro_atlas_raster = self.hydro_atlas_embed(hydroatlas_inp)
            context_statics_lst.append(embedded_hydro_atlas_raster)
        if self.w_dem_context:
            context_statics_lst.append(dem_context_inp)
        # Add the statics in at every step
        if self.w_hydroatlas_basin or self.w_dem_context:
            context_statics = (
                torch.cat(context_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
            )
            context_inp = torch.cat([era5l_inp, era5_inp, context_statics], dim=2)
        else:
            context_inp = torch.cat([era5l_inp, era5_inp], dim=2)
        context_embedded = self.context_embed(context_inp, batch_positions)
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

        # Get Lead time embedding indexes
        if self.w_s1:
            lead_idx = self.get_lead_time_idx(ex["s1_lead_days"])
            lead = self.lead_time_embedding(lead_idx)
        else:
            lead = None

        # Process local inputs
        local_lst = [context_out_upsc]
        if self.w_s1:
            local_lst.append(s1_inp)
        if self.w_dem_local:
            local_lst.append(dem_local_inp)
        if self.w_hand:
            local_lst.append(hand_inp)
        local_inp = torch.cat(local_lst, dim=1)
        # Pretend it's temporal data with one time step for utae
        local_inp = local_inp[:, None]
        out = self.local_embed(local_inp, batch_positions[:, :1], lead=lead)
        out = out[:, 0]
        return out
    
class Two3DUNet(nn.Module):
    def __init__(
        self,
        era5_bands,
        era5l_bands,
        hydroatlas_bands=[],
        hydroatlas_dim=None,
        lead_time_dim=None,
        norms={},
        w_hydroatlas_basin=True,
        w_dem_context=True,
        w_dem_local=False,
        w_hand=True,
        w_s1=True,
        n_predict=3,
        weather_window_size=20,
        context_embed_output_dim=5,
        center_crop_context=True,
        average_context=True,
    ):
        super().__init__()
        self.era5_bands = era5_bands
        self.era5l_bands = era5l_bands
        self.hydroatlas_bands = hydroatlas_bands
        self.w_hydroatlas_basin = w_hydroatlas_basin
        self.w_dem_context = w_dem_context
        self.w_dem_local = w_dem_local
        self.w_hand = w_hand
        self.w_s1 = w_s1
        self.n_predict = n_predict
        self.weather_window_size = weather_window_size
        self.center_crop_context = center_crop_context
        self.average_context = average_context

        # Store normalisation info on model
        # (To load model weights, the shapes must be identical; so use empty if not known at init)
        empty_norms = get_empty_norms(len(era5_bands), len(era5l_bands), len(hydroatlas_bands))
        for key in ["era5", "era5_land", "hydroatlas_basin", "dem", "s1", "hand"]:
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

        # Create context embedding layers
        self.n_weather = len(era5_bands) + len(era5l_bands)
        self.n_hydroatlas = len(hydroatlas_bands)
        context_embed_input_dim = self.n_weather
        if self.w_hydroatlas_basin:
            self.hydro_atlas_embed = nn.Conv2d(
                self.n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1
            )
            context_embed_input_dim += hydroatlas_dim
        if self.w_dem_context:
            context_embed_input_dim += 1
        self.context_embed = unet3d.UNet3D(
            input_dim=context_embed_input_dim,
            out_conv=[context_embed_output_dim],
        )

        # Create local embedding/prediction layers
        local_input_dim = context_embed_output_dim
        if self.w_dem_local:
            local_input_dim += 1
        if self.w_hand:
            local_input_dim += 1
        if self.w_s1:
            local_input_dim += 2
            self.len_lead = weather_window_size + 1
            self.lead_time_embedding = nn.Embedding(self.len_lead, lead_time_dim)
        self.local_embed = unet3d.UNet3D(
            input_dim=local_input_dim,
            out_conv=[64, n_predict],
            cond_dim=lead_time_dim,
        )

    def normalise(self, ex, key, suffix=None):
        if suffix is not None:
            ex_key = f"{key}_{suffix}"
        else:
            ex_key = key
        if ex_key in ex:
            data = ex[ex_key]
            km = f"{key}_mean"
            ks = f"{key}_std"
            data = (data - getattr(self, km)) / getattr(self, ks)
            return data

    def get_lead_time_idx(self, lead):
        # Get the index into the embedding based on lead.
        # For lead times within the weather window size, lead == idx
        lead_copy = lead.clone()
        # For leads too far away in time we just set it to the last idx
        # This allows the model to know how to use soil moisture in the S1 images,
        # regardless of how old. e.g. the model can ignore it if it wants.
        lead_copy[lead > self.len_lead] = self.len_lead - 1
        return lead_copy

    def forward(self, ex):
        B, N, cC, cH, cW = ex["era5"].shape
        batch_positions = torch.arange(0, N).reshape((1, N)).repeat((B, 1)).to(ex["era5"].device)
        if self.w_s1:
            example_local = ex["s1"]
        else:
            example_local = ex["dem_local"]
        fH, fW = example_local.shape[-2:]

        # Normalise inputs
        era5_inp = self.normalise(ex, "era5")
        era5l_inp = self.normalise(ex, "era5_land")
        hydroatlas_inp = self.normalise(ex, "hydroatlas_basin")
        dem_context_inp = self.normalise(ex, "dem", "context")
        dem_local_inp = self.normalise(ex, "dem", "local")
        hand_inp = self.normalise(ex, "hand")
        s1_inp = self.normalise(ex, "s1")

        # These inputs might have nan
        era5l_inp = nans_to_zero(era5l_inp)
        hydroatlas_inp = nans_to_zero(hydroatlas_inp)
        dem_context_inp = nans_to_zero(dem_context_inp)
        dem_local_inp = nans_to_zero(dem_local_inp)
        hand_inp = nans_to_zero(hand_inp)

        # Process context inputs
        context_statics_lst = []
        if self.w_hydroatlas_basin:
            embedded_hydro_atlas_raster = self.hydro_atlas_embed(hydroatlas_inp)
            context_statics_lst.append(embedded_hydro_atlas_raster)
        if self.w_dem_context:
            context_statics_lst.append(dem_context_inp)
        # Add the statics in at every step
        if self.w_hydroatlas_basin or self.w_dem_context:
            context_statics = (
                torch.cat(context_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
            )
            context_inp = torch.cat([era5l_inp, era5_inp, context_statics], dim=2)
        else:
            context_inp = torch.cat([era5l_inp, era5_inp], dim=2)
        context_embedded = self.context_embed(context_inp)
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

        # Get Lead time embedding indexes
        if self.w_s1:
            lead_idx = self.get_lead_time_idx(ex["s1_lead_days"])
            lead = self.lead_time_embedding(lead_idx)
        else:
            lead = None

        # Process local inputs
        local_lst = [context_out_upsc]
        if self.w_s1:
            local_lst.append(s1_inp)
        if self.w_dem_local:
            local_lst.append(dem_local_inp)
        if self.w_hand:
            local_lst.append(hand_inp)
        local_inp = torch.cat(local_lst, dim=1)
        # Pretend it's temporal data with one time step for utae
        local_inp = local_inp[:, None]
        out = self.local_embed(local_inp, lead=lead)
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
    lead_time_dim = 16
    ex = {
        "era5": torch.randn((B, T, n_era5, cH, cW)).cuda(),
        "era5_land": torch.randn((B, T, n_era5_land, cH, cW)).cuda(),
        "hydroatlas_basin": torch.randn((B, n_hydroatlas, cH, cW)).cuda(),
        "dem_context": torch.randn((B, 1, cH, cW)).cuda(),
        "s1": torch.randn(B, 2, fH, fW).cuda(),
        "dem_local": torch.randn((B, 1, fH, fW)).cuda(),
        "hand": torch.randn((B, 1, fH, fW)).cuda(),
        "s1_lead_days": torch.randint(0, 20, (B,)).cuda(),
    }
    era5_bands = list(range(n_era5))
    era5l_bands = list(range(n_era5_land))
    hydroatlas_bands = list(range(n_hydroatlas))
    model = TwoUTAE(
        era5_bands,
        era5l_bands,
        hydroatlas_bands,
        hydroatlas_dim,
        lead_time_dim,
    )
    model = model.cuda()
    model1 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
    )
    model1 = model1.cuda()
    model2 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
    )
    model2 = model2.cuda()
    model3 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
    )
    model3 = model3.cuda()
    model4 = TwoUTAE(
        era5_bands,
        era5l_bands,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_s1=False,
    )
    model4 = model4.cuda()
    model5 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_dem_local=False,
    )
    model5 = model5.cuda()
    model6 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_dem_local=False,
        average_context=False,
    )
    model6 = model6.cuda()
    model7 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_dem_local=False,
        center_crop_context=False,
        average_context=False,
    )
    model7 = model7.cuda()
    print("Model")
    out = model(ex)
    # Only hydroatlas will cause problems if provided after saying we wouldn't
    ex.pop("hydroatlas_basin")
    print("Model 1")
    out = model1(ex)
    print("Model 2")
    out = model2(ex)
    print("Model 3")
    out = model3(ex)
    print("Model 4")
    out = model4(ex)
    print("Model 5")
    out = model5(ex)
    print("Model 6")
    out = model6(ex)
    print("Model 7")
    out = model7(ex)
    print(out.shape)
