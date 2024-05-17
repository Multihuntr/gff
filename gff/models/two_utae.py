import torch
import torch.nn as nn

# from . import utae
import gff.models.utae as utae


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
        w_dem_local=True,
        w_s1=True,
        n_predict=3,
        weather_window_size=20,
        context_embed_output_dim=3,
        temp_encoding="ltae"
    ):
        super().__init__()
        self.era5_bands = era5_bands
        self.era5l_bands = era5l_bands
        self.hydroatlas_bands = hydroatlas_bands
        self.w_hydroatlas_basin = w_hydroatlas_basin
        self.w_dem_context = w_dem_context
        self.w_dem_local = w_dem_local
        self.w_s1 = w_s1
        self.n_predict = n_predict
        self.weather_window_size = weather_window_size
        self.temp_encoding = temp_encoding

        # Store normalisation info on model
        # (To load model weights, the shapes must be identical; so use empty if not known at init)
        empty_norms = get_empty_norms(len(era5_bands), len(era5l_bands), len(hydroatlas_bands))
        for key in ["era5", "era5_land", "hydroatlas_basin", "dem", "s1"]:
            if key in norms:
                mean, std = norms[key]
            else:
                mean, std = empty_norms[key]
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std", std)

        assert (
            self.w_s1 or self.w_dem_local
        ), "Must provide either s1 or dem local to produce local scale predictions"
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
            temp_encoding=self.temp_encoding
        )

        # Create local embedding/prediction layers
        local_input_dim = context_embed_output_dim
        if self.w_dem_local:
            local_input_dim += 1
        if self.w_s1:
            local_input_dim += 2
            self.lead_m = 3
            self.lead_n = 5  # see self.get_lead_time_idx for details
            self.len_lead = (
                weather_window_size + self.lead_m * weather_window_size // self.lead_n + 1
            )
            self.lead_time_embedding = nn.Embedding(self.len_lead, lead_time_dim)
        self.local_embed = utae.UTAE(
            local_input_dim,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[64, 64, 64, 128],
            out_conv=[64, n_predict],
            cond_dim=lead_time_dim,
            temp_encoding=self.temp_encoding
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
        # For normal FiLM conditioning, lead == idx, but here we're encoding an unbounded value.
        # So, we apply a simple remapping scheme for values outside the weather window.
        # 1. For lead times within the weather window size, lead == idx
        lead_copy = lead.clone()
        # 2. For lead times outside the weather window, we chunk by self.lead_n days
        outside_window = lead > self.weather_window_size
        chunked_idx = (lead[outside_window] - self.weather_window_size) // self.lead_n
        lead_copy[outside_window] = chunked_idx + self.weather_window_size
        # 3. For leads too far away in time (lead_m * window size) we just set it to the last idx
        # This allows the model to know how to use soil moisture in the S1 images,
        # regardless of how old. e.g. the model can ignore it if it wants.
        lead_copy[lead > (self.lead_m * self.weather_window_size)] = self.len_lead - 1
        return lead_copy

    def forward(self, ex):
        B, N, cC, cH, cW = ex["era5_land"].shape
        batch_positions = (
            torch.arange(0, N).reshape((1, N)).repeat((B, 1)).to(ex["era5_land"].device)
        )
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
        s1_inp = self.normalise(ex, "s1")

        # These inputs might have nan
        era5l_inp = nans_to_zero(era5l_inp)
        hydroatlas_inp = nans_to_zero(hydroatlas_inp)
        dem_context_inp = nans_to_zero(dem_context_inp)
        dem_local_inp = nans_to_zero(dem_local_inp)

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
        ylo, yhi = cH // 2 - 1, cH // 2 + 3
        xlo, xhi = cW // 2 - 1, cW // 2 + 3
        context_out = context_embedded[:, :, ylo:yhi, xlo:xhi].mean(axis=(2, 3), keepdims=True)
        context_out_repeat = context_out.repeat((1, 1, fH, fW))

        # Get Lead time embedding indexes
        if self.w_s1:
            lead_idx = self.get_lead_time_idx(ex["s1_lead_days"])
            lead = self.lead_time_embedding(lead_idx)
        else:
            lead = None

        # Process local inputs
        local_lst = [context_out_repeat]
        if self.w_s1:
            local_lst.append(s1_inp)
        if self.w_dem_local:
            local_lst.append(dem_local_inp)
        local_inp = torch.cat(local_lst, dim=1)
        # Pretend it's temporal data with one time step for utae
        local_inp = local_inp[:, None]
        out = self.local_embed(local_inp, batch_positions[:, :1], lead=lead)
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
        "s1_lead_days": torch.randint(0, 20, (B,)).cuda(),
    }
    era5_bands = list(range(n_era5))
    era5l_bands = list(range(n_era5_land))
    hydroatlas_bands = list(range(n_hydroatlas))
    model = TwoUTAE(era5_bands, era5l_bands, hydroatlas_bands, hydroatlas_dim, lead_time_dim,temp_encoding='ltae')
    model = model.cuda()
    model1 = TwoUTAE(
        era5_bands, era5l_bands, lead_time_dim=lead_time_dim, w_hydroatlas_basin=False,temp_encoding='ltae'
    )
    model1 = model1.cuda()
    model2 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        temp_encoding='ltae'
    )
    model2 = model2.cuda()
    model3 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        temp_encoding='ltae'
    )
    model3 = model3.cuda()
    model4 = TwoUTAE(
        era5_bands, era5l_bands, w_hydroatlas_basin=False, w_dem_context=False, w_s1=False,temp_encoding='ltae'
    )
    model4 = model4.cuda()
    model5 = TwoUTAE(
        era5_bands,
        era5l_bands,
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_dem_local=False,
        temp_encoding='ltae'
    )
    model5 = model5.cuda()
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
    print(out.shape)
