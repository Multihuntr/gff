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
                ex["dem_coarse"],
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
        inp_fineres = torch.cat([s1, ex["dem_fine"]], axis=1)
        fine_features = self.fine_to_bottleneck(inp_fineres, coarse_global_embed[-1])

        # Finally build out from bottleneck to segmentation logits
        logits = self.bottleneck_to_fine(fine_features[-1], fine_features[:-1])

        return logits


class DebugModel(nn.Module):
    def __init__(self, n_hydroatlas, hydroatlas_dim, n_coarse, n_fine, n_predict=3):
        self.hydro_atlas_embed = nn.Conv2d(n_hydroatlas, hydroatlas_dim, kernel_size=3, padding=1)
        self.coarse_embed = utae.UTAE(
            n_coarse, encoder_widths=[32, 32], decoder_widths=[32, 32], encoder=True
        )
        self.coarse_project = nn.Conv2d(32, n_fine)
        self.fine_embed = utae.UTAE(n_fine * 2, out_conv=[32, n_predict])

    def forward(self, ex):
        B, N, cC, cH, cW = ex["era5_land"].shape
        B, fC, fH, fW = ex["s1"].shape[-3:]

        # Process coarse inputs
        embedded_hydro_atlas_raster = self.hydro_atlas_embed(ex["hydro_basin_raster"])
        coarse_statics_lst = [
            embedded_hydro_atlas_raster,
            ex["hydro_river_raster"],
            ex["dem_coarse"],
        ]
        coarse_statics = torch.cat(coarse_statics_lst, dim=1).unsqueeze(1).repeat((1, N, 1, 1, 1))
        coarse_inp = torch.cat([ex["era5_land"], coarse_statics], dim=2)
        coarse_embedded = self.coarse_project(self.coarse_embed(coarse_inp))

        # Select the central 2x2 pixels and average
        ylo, yhi = cH // 2 - 1, cH // 2 + 3
        xlo, xhi = cW // 2 - 1, cW // 2 + 3
        coarse_out = coarse_embedded[:, :, ylo:yhi, xlo:xhi].mean(axis=(3, 4), keepdims=True)
        coarse_out_repeat = coarse_out.repeat((1, 1, fH, fW))

        # Process fine inputs
        s1_inps = [ex["s1"][:, i] for i in range(fC)]
        fine_inp = torch.cat([coarse_out_repeat, *s1_inps, ex["dem_fine"]], dim=1)
        out = self.fine_embed(fine_inp)
        return out
