import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utae


class SequentialAdd(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.lyrs = nn.ModuleList(layers)

    def forward(self, inp, to_add):
        count = 0
        outs = [inp]
        for lyr in self.lyrs:
            out = outs[-1]
            if isinstance(lyr, nn.Conv2d):
                if isinstance(to_add, list):
                    out = out + to_add[count]
                    count += 1
                else:
                    out = out + to_add
            out = lyr(out)
            outs.append(out)
        return outs


class TemporalEncoderAdd(nn.Module):
    pass


class UpsampleConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, padding=1)

    def forward(self, inp):
        return self.conv(F.interpolate(inp, size=2, mode="bilinear"))


class ContextInjection(nn.Module):
    def __init__(self, n_hydroatlas, hydroatlas_dim, n_coarse, n_fine, bottleneck_dim, n_predict):
        super().__init__()
        self.hydro_atlas_embed = nn.Conv2d(n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1)

        self.coarse_embed = nn.Sequential(
            nn.Conv2d(n_coarse, bottleneck_dim / 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 4, bottleneck_dim / 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Conv2d(bottleneck_dim / 4, bottleneck_dim / 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 2, bottleneck_dim, kernel_size=4, stride=2),
        )

        self.fine_to_bottleneck = SequentialAdd(
            nn.Conv2d(n_fine, bottleneck_dim / 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 4, bottleneck_dim / 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 4, bottleneck_dim / 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 2, bottleneck_dim / 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim / 2, bottleneck_dim, kernel_size=4, stride=2),
        )

        self.bottleneck_to_fine = nn.Sequential(
            SequentialAdd(
                UpsampleConv(bottleneck_dim, bottleneck_dim / 2, kernel_size=3, padding=1),
                nn.ReLU(),
                UpsampleConv(bottleneck_dim / 2, bottleneck_dim / 2, kernel_size=3, padding=1),
                nn.ReLU(),
                UpsampleConv(bottleneck_dim / 2, bottleneck_dim / 4, kernel_size=3, padding=1),
                nn.ReLU(),
                UpsampleConv(bottleneck_dim / 4, bottleneck_dim / 4, kernel_size=3, padding=1),
                nn.ReLU(),
                UpsampleConv(bottleneck_dim / 4, bottleneck_dim / 4, kernel_size=3, padding=1),
            ),
            nn.Conv2d(bottleneck_dim / 4, n_predict, kernel_size=3, padding=1),
        )

        self.coarse_dynamic_embed = TemporalEncoderAdd(
            (bottleneck_dim + n_coarse), bottleneck_dim, n_layers=2, n_heads=8
        )

    def forward(self, ex):
        # Process coarse data to a global embedding
        embedded_hydro_atlas_raster = self.hydro_atlas_embed(ex["hydro_basin_raster"])
        coarse_statics = torch.cat(
            [
                embedded_hydro_atlas_raster,
                ex["hydro_river_raster"],
                ex["dem_context"],
            ],
            dim=2,
        )
        B, T = ex["era5_land"].shape[:2]
        position_encoding = self.position_encoder(T)
        era5_land_bxt = einops.rearrange(ex["era5_land"], "b t c h w -> (b t) c h w")
        coarse_independent_bxt = self.coarse_dynamic_embed(
            era5_land_bxt, add_many=coarse_statics, add_each=position_encoding
        )
        coarse_independent = einops.rearrange(coarse_independent_bxt, "(b t) c -> b t c", b=B, t=T)
        coarse_global_embed = self.coarse_temporal_embed(coarse_independent)

        # Process fine data up to the bottleneck
        s1 = einops.rearrange(ex["s1"], "b t c h w -> b (t c) h w")
        inp_fineres = torch.cat([s1, ex["dem_local"]], axis=1)
        fine_features = self.fine_to_bottleneck(inp_fineres, coarse_global_embed[-1])

        # Finally build out from bottleneck to segmentation logits
        logits = self.bottleneck_to_fine(fine_features[-1], fine_features[:-1])

        return logits


class DebugModel(nn.Module):
    def __init__(self, n_hydroatlas, hydroatlas_dim, n_context, n_predict=3):
        super().__init__()
        self.hydro_atlas_embed = nn.Conv2d(n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1)
        self.context_embed = utae.UTAE(
            n_context + hydroatlas_dim + 1,
            encoder_widths=[32, 32],
            decoder_widths=[32, 32],
            out_conv=[3],
        )
        self.fine_embed = utae.UTAE(6, out_conv=[32, n_predict])

    def forward(self, ex):
        B, N, cC, cH, cW = ex["era5_land"].shape
        fC, fH, fW = ex["s1"].shape[-3:]
        batch_positions = torch.arange(0, N).reshape((1, N)).repeat((B, 1)).to(ex["s1"].device)

        # Process context inputs
        embedded_hydro_atlas_raster = self.hydro_atlas_embed(ex["hydroatlas_basin"])
        context_statics_lst = [embedded_hydro_atlas_raster, ex["dem_context"]]
        # Add the statics in at every step
        context_statics = (
            torch.cat(context_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
        )
        context_inp = torch.cat([ex["era5_land"], ex["era5"], context_statics], dim=2)
        context_embedded = self.context_embed(context_inp, batch_positions)

        # Select the central 2x2 pixels and average
        ylo, yhi = cH // 2 - 1, cH // 2 + 3
        xlo, xhi = cW // 2 - 1, cW // 2 + 3
        context_out = context_embedded[:, :, ylo:yhi, xlo:xhi].mean(axis=(2, 3), keepdims=True)
        context_out_repeat = context_out.repeat((1, 1, fH, fW))

        # Process fine inputs
        fine_inp = torch.cat([context_out_repeat, ex["s1"], ex["dem_local"]], dim=1)
        # Pretend it's temporal data with one time step for utae
        fine_inp = fine_inp[:, None]
        out = self.fine_embed(fine_inp, batch_positions[:, :1])
        return out


if __name__ == "__main__":
    B, T = 8, 12
    cH, cW = 32, 32
    fH, fW = 224, 224
    n_hydroatlas = 128
    hydroatlas_dim = 16
    n_era5 = 16
    n_era5_land = 8
    ex = {
        "era5": torch.randn((B, T, n_era5, cH, cW)).cuda(),
        "era5_land": torch.randn((B, T, n_era5_land, cH, cW)).cuda(),
        "hydroatlas_basin": torch.randn((B, n_hydroatlas, cH, cW)).cuda(),
        "dem_context": torch.randn((B, 1, cH, cW)).cuda(),
        "s1": torch.randn(B, 2, fH, fW).cuda(),
        "dem_local": torch.randn((B, 1, fH, fW)).cuda(),
    }
    model = DebugModel(n_hydroatlas, hydroatlas_dim, (n_era5 + n_era5_land))
    model = model.cuda()
    out = model(ex)
    print(out.shape)
