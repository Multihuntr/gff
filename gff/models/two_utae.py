import torch
import torch.nn as nn

from . import utae


class TwoUTAE(nn.Module):
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
    model = TwoUTAE(n_hydroatlas, hydroatlas_dim, (n_era5 + n_era5_land))
    model = model.cuda()
    out = model(ex)
    print(out.shape)
