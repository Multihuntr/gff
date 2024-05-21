"""
Taken from https://github.com/VSainteuf/utae-paps and modified to suit our needs
They took it themselves from https://github.com/roserustowicz/crop-type-mapping/
Implementation by the authors of the paper :
"Semantic Segmentation of crop type in Africa: A novel Dataset and analysis of deep learning methods"
R.M. Rustowicz et al.
"""

import torch
import torch.nn as nn


def conv_block(in_dim, middle_dim, out_dim, cond_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model

def center_in(in_dim, out_dim, cond_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_out(in_dim, out_dim, cond_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
    )
    return model


def up_conv_block(in_dim, out_dim, cond_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model

##################

class ConvLayer3d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        cond_dim=None,  # lead_time_embed_dim
        pool=False
    ):
        super(ConvLayer3d, self).__init__()

        self.mlp = None  # TODO: FiLM

        layers = []       
        layers.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(out_dim))
        layers.append(nn.LeakyReLU(inplace=True))
        # if pool:
        #     layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.conv = nn.Sequential(*layers)

        if exists(cond_dim):
            self.mlp = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(cond_dim, out_dim * 2),
                        ).to("cuda")

    def forward(self, input, lead=None):
        assert not (exists(self.mlp) ^ exists(lead))
        scale_shift = None
        if exists(self.mlp) and exists(lead):
            lead = self.mlp(lead)
            lead = rearrange(lead.flatten(start_dim=1), "b c -> b c 1 1")
            scale_shift = lead.chunk(2, dim=1)

        for _, layer in enumerate(self.conv):
            input = self.conv(input)
            isNormLayer = isinstance(layer, nn.BatchNorm3d)
            # apply linear transform on CONV output
            # (following a normalization, and preceding an output non-linearity)
            if (
                exists(scale_shift)
                and isNormLayer
            ):
                scale, shift = scale_shift
                replicate_each_t = int(input.shape[0] / shift.shape[0])
                input = input * (scale.repeat(replicate_each_t, 1, 1, 1) + 1) + shift.repeat(
                    replicate_each_t, 1, 1, 1
                )
        return input  # self.conv(input, lead)
    

class ConvTransposeLayer3d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        cond_dim=None,  # lead_time_embed_dim
        block=True,
    ):
        super(ConvTransposeLayer3d, self).__init__()

        self.mlp = None  # TODO: FiLM

        layers = []    
        self.norm_then_relu = []    
        layers.append(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
        if block:
            layers.append(nn.BatchNorm3d(out_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            uses_relu = True
            self.norm_then_relu.append(uses_relu)
        self.conv = nn.Sequential(*layers)

        if exists(cond_dim):
            self.mlp = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(cond_dim, out_dim * 2),
                        ).to("cuda")

    def forward(self, input, lead=None):
        assert not (exists(self.mlp) ^ exists(lead))
        scale_shift = None
        if exists(self.mlp) and exists(lead):
            lead = self.mlp(lead)
            lead = rearrange(lead.flatten(start_dim=1), "b c -> b c 1 1")
            scale_shift = lead.chunk(2, dim=1)

        for _, layer in enumerate(self.conv):
            input = self.conv(input)
            isNormLayer = isinstance(layer, nn.BatchNorm3d)
            # apply linear transform on CONV output
            # (following a normalization, and preceding an output non-linearity)
            if (
                exists(scale_shift)
                and isNormLayer
            ):
                scale, shift = scale_shift
                replicate_each_t = int(input.shape[0] / shift.shape[0])
                input = input * (scale.repeat(replicate_each_t, 1, 1, 1) + 1) + shift.repeat(
                    replicate_each_t, 1, 1, 1
                )
        return input  # self.conv(input, lead)


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dims,
        cond_dim=None,  # lead_time_embed_dim
        end_pool=True,
        skip=True
    ):
        super(ConvBlock3d, self).__init__()
        self.skip=skip
        self.end_pool=end_pool
        layers=[]
        for i in range(len(out_dims)):
            if i >0:
                layers.append(
                    ConvLayer3d(
                    in_dim==out_dims[i-1],
                    out_dim=out_dims[i],
                    cond_dim=cond_dim,
                    pool=True
                ))
            else:
                layers.append(
                    ConvLayer3d(
                    in_dim==in_dim,
                    out_dim=out_dims[i],
                    cond_dim=cond_dim,
                    pool=True
                ))
        self.conv=nn.ModuleList(*layers)
        if self.end_pool:
            self.pool=nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


    def forward(self, input, lead=None):
        feats = self.conv(input, lead)
        if self.end_pool:
            feats_down = self.pool(feats)
        if self.skip:
            return feats, feats_down
        else:
            return feats_down
    

class CenterConvBlock3d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dims,
        cond_dim=None,  # lead_time_embed_dim
    ):
        super(ConvBlock3d, self).__init__()

        layers=[]
        for i in range(len(out_dims)-1):
            if i >0:
                layers.append(
                    ConvLayer3d(
                    in_dim==out_dims[i-1],
                    out_dim=out_dims[i],
                    cond_dim=cond_dim,
                ))
            else:
                layers.append(
                    ConvLayer3d(
                    in_dim==in_dim,
                    out_dim=out_dims[i],
                    cond_dim=cond_dim,
                ))
        self.conv=nn.ModuleList(*layers)
        self.out = nn.ConvTranspose3d(out_dims[-2], out_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, input, lead=None):
        center = self.conv(input, lead)
        return self.out(center)


##################


class UNet3D(nn.Module):
    def __init__(
        self, 
        input_dim, 
        out_conv, 
        feats=8, 
        pad_value=None, 
        zero_pad=True, 
        cond_dim=None
    ):
        super(UNet3D, self).__init__()
        self.input_dim = input_dim
        self.out_conv = out_conv
        self.pad_value = pad_value
        self.zero_pad = zero_pad

        self.encoder1 = ConvBlock3d(
            in_dim=input_dim, 
            out_dims=[feats*4,feats*4], 
            cond_dim=cond_dim,
            end_pool=True,
            skip=True
        )
        self.encoder2 = ConvBlock3d(
            in_dim=feats*4, 
            out_dims=[feats*8, feats*8], 
            cond_dim=cond_dim,
            end_pool=True,
            skip=True
        )

        out_dims_center = [feats*16, feats*16, feats*8]
        self.center = CenterConvBlock3d(
            in_dim=feats*8,
            out_dims=out_dims_center,
            cond_dim=cond_dim
        )

        decoder_layers1 = []
        decoder_layers1.append(
            ConvBlock3d(
                in_dim=feats*16,
                out_dims=[feats*8, feats*8],
                cond_dim=cond_dim,
                end_pool=False,
                skip=False
            )
        )
        decoder_layers1.append(
            ConvTransposeLayer3d(
                in_dim=feats*8,
                out_dim=feats*4,
                cond_dim=cond_dim,
                block=True
            )
        )
        self.decoder1 = nn.ModuleList(*decoder_layers1)
        decoder_layers2 = []
        decoder_layers2.append(
            ConvBlock3d(
                in_dim=feats*8,
                out_dims=[feats*4, feats*2],
                cond_dim=cond_dim,
                end_pool=False,
                skip=False
            )
        )
        decoder_layers2.append(
            nn.Conv3d(
                feats * 2, out_conv, kernel_size=3, stride=1, padding=1
            )
        )
        self.decoder2 = nn.ModuleList(*decoder_layers2)


    def forward(self, x, lead=None):
        out = x.permute(0, 2, 1, 3, 4)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask
            if self.zero_pad:
                out[out == self.pad_value] = 0
        skip1, feats_down = self.encoder1(out, lead)
        skip2, feats_down = self.encoder2(feats_down, lead)
        center = self.center(feats_down, lead)
        concat2 = torch.cat([center, skip2[:, :, :center.shape[2], :, :]], dim=1)
        feats_up = self.decoder1(concat2)
        concat1 = torch.cat([feats_up, skip1[:, :, :feats_up.shape[2], :, :]], dim=1)
        final = self.decoder2(concat1)
        final = final.permute(0, 1, 3, 4, 2)  # BxCxHxWxT
        if self.pad_value is not None:
            if pad_mask.any():
                # masked mean
                pad_mask = pad_mask[:, :final.shape[-1]] #match new temporal length (due to pooling)
                pad_mask = ~pad_mask # 0 on padded values
                out = (final.permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(
                    dim=-1)[None, None, None, :]
                out = out.permute(3, 0, 1, 2)
            else:
                out = final.mean(dim=-1)
        else:
            out = final.mean(dim=-1)

        return out
