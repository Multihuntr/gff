import torch
import torch.nn as nn

from . import utae


class TwoUTAE(nn.Module):
    def __init__(
        self,
        n_weather,
        n_hydroatlas=None,
        hydroatlas_dim=None,
        lead_time_dim=None,
        w_hydroatlas_basin=True,
        w_dem_context=True,
        w_dem_local=True,
        w_s1=True,
        n_predict=3,
        weather_window_size=20,
        context_embed_output_dim=3,
    ):
        super().__init__()
        self.w_hydroatlas_basin = w_hydroatlas_basin
        self.w_dem_context = w_dem_context
        self.w_dem_local = w_dem_local
        self.w_s1 = w_s1
        self.weather_window_size = weather_window_size

        assert (
            self.w_s1 or self.w_dem_local
        ), "Must provide either s1 or dem local to produce local scale predictions"
        assert (self.w_s1 and (lead_time_dim is not None)) or (
            (not self.w_s1) and (lead_time_dim is None)
        ), "If you provide s1, you must also provide lead_time_dim. If not, don't provide it."

        # Create context embedding layers
        context_embed_input_dim = n_weather
        if self.w_hydroatlas_basin:
            self.hydro_atlas_embed = nn.Conv2d(
                n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1
            )
            context_embed_input_dim += hydroatlas_dim
        if self.w_dem_context:
            context_embed_input_dim += 1
        self.context_embed = utae.UTAE(
            context_embed_input_dim,
            encoder_widths=[32, 32],
            decoder_widths=[32, 32],
            out_conv=[context_embed_output_dim],
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
        )

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

        # Process context inputs
        context_statics_lst = []
        if self.w_hydroatlas_basin:
            embedded_hydro_atlas_raster = self.hydro_atlas_embed(ex["hydroatlas_basin"])
            context_statics_lst.append(embedded_hydro_atlas_raster)
        if self.w_dem_context:
            context_statics_lst.append(ex["dem_context"])
        # Add the statics in at every step
        if self.w_hydroatlas_basin or self.w_dem_context:
            context_statics = (
                torch.cat(context_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
            )
            context_inp = torch.cat([ex["era5_land"], ex["era5"], context_statics], dim=2)
        else:
            context_inp = torch.cat([ex["era5_land"], ex["era5"]], dim=2)
        print("context:", context_inp.shape)
        context_embedded = self.context_embed(context_inp, batch_positions)

        # Select the central 2x2 pixels and average
        ylo, yhi = cH // 2 - 1, cH // 2 + 3
        xlo, xhi = cW // 2 - 1, cW // 2 + 3
        context_out = context_embedded[:, 0, :, ylo:yhi, xlo:xhi].mean(axis=(2, 3), keepdims=True)
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
            local_lst.append(ex["s1"])
        if self.w_dem_local:
            local_lst.append(ex["dem_local"])
        local_inp = torch.cat(local_lst, dim=1)
        # Pretend it's temporal data with one time step for utae
        local_inp = local_inp[:, None]
        print("local:  ", local_inp.shape)
        out = self.local_embed(local_inp, batch_positions[:, :1], lead=lead)
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
    model = TwoUTAE((n_era5 + n_era5_land), n_hydroatlas, hydroatlas_dim, lead_time_dim)
    model = model.cuda()
    model1 = TwoUTAE((n_era5 + n_era5_land), lead_time_dim=lead_time_dim, w_hydroatlas_basin=False)
    model1 = model1.cuda()
    model2 = TwoUTAE(
        (n_era5 + n_era5_land),
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
    )
    model2 = model2.cuda()
    model3 = TwoUTAE(
        (n_era5 + n_era5_land),
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
    )
    model3 = model3.cuda()
    model4 = TwoUTAE(
        (n_era5 + n_era5_land), w_hydroatlas_basin=False, w_dem_context=False, w_s1=False
    )
    model4 = model4.cuda()
    model5 = TwoUTAE(
        (n_era5 + n_era5_land),
        lead_time_dim=lead_time_dim,
        w_hydroatlas_basin=False,
        w_dem_context=False,
        w_dem_local=False,
    )
    model5 = model5.cuda()
    print("Model")
    out = model(ex)
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
