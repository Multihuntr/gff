"""
U-TAE Implementation
Original Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
Edited by: Patrick Ebel
Copied by: Me!
Source: https://github.com/mondlichtpierrot/coastalFlood/blob/master/models/UnCRtainTS/model/src/backbones/utae.py
"""

import torch
import torch.nn as nn

import numpy as np
from einops import rearrange

from .convlstm import ConvLSTM, BConvLSTM
from .ltae import LTAE2d


# function to normalize gradient magnitudes,
#   evoke via e.g. scale_gradients(out) at every forward pass
def scale_gradients(params):
    def hook_norm(grad):
        # get norm of parameter p's gradients
        # grad_norm = p.grad.detach().data.norm(2)
        # get the gradient's L2 norm
        grad_norm = grad.detach().data.norm(2)
        # return normalized gradient
        return grad / (grad_norm + 1e-9)

    # see https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
    params.register_hook(hook_norm)


def exists(val):
    return val is not None


class UNet(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[2],
        out_nonlin_mean=False,
        out_nonlin_var="relu",
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        encoder_norm="group",
        norm_skip="batch",
        norm_up="batch",
        decoder_norm="batch",
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):
        """
        U-Net architecture for spatial pre-training of UTAE on mono-temporal data, excluding LTAE temporal encoder.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
                - none: apply no normalization
            norm_skip (str): similar to encoder_norm, just controlling the normalization after convolving skipped maps
            norm_up (str): similar to encoder_norm, just controlling the normalization after transposed convolution
            decoder_norm (str): similar to encoder_norm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(UNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        # ENCODER
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0]],
            k=1,
            s=1,
            p=0,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        # DECODER
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm_skip=norm_skip,  #'batch'
                norm_up=norm_up,  # 'batch'
                norm=decoder_norm,  # "batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        # note: not including normalization layer and ReLU nonlinearity into the final ConvBlock,
        #       if inserting >1 layers into out_conv then consider treating normalizations separately
        self.out_dims = out_conv[-1]
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv,
            k=1,
            s=1,
            p=0,
            padding_mode=padding_mode,
            norm="none",
            last_relu=False,
        )

        if out_nonlin_mean:
            self.out_mean = nn.Sigmoid()  # this is for predicting mean values in [0, 1]
        else:
            self.out_mean = (
                nn.Identity()
            )  # just keep the mean estimates, without applying a nonlinearity

        if out_nonlin_var == "relu":
            self.out_var = nn.ReLU()  # this is for predicting var values > 0
        elif out_nonlin_var == "softplus":
            self.out_var = nn.Softplus(
                beta=1, threshold=20
            )  # a smooth approximation to the ReLU function
        elif out_nonlin_var == "elu":
            self.out_var = lambda vars: nn.ELU()(vars) + 1 + 1e-8
        else:  # just keep the variance estimates,
            self.out_var = (
                nn.Identity()
            )  # just keep the variance estimates, without applying a nonlinearity

    def forward(self, input, batch_positions=None, return_att=False):
        # SPATIAL ENCODER
        # collect feature maps in list 'feature_maps'
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        out = out[:, 0, ...]  # note: we index to reduce the temporal dummy dimension of size 1
        for i in range(self.n_stages - 1):
            # skip-connect features between paired encoder/decoder blocks
            skip = feature_maps[-(i + 2)]
            # upconv the features, concatenating current 'out' and paired 'skip'
            out = self.up_blocks[i](
                out, skip[:, 0, ...]
            )  # note: we index to reduce the temporal dummy dimension of size 1
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            # append a singelton temporal dimension such that outputs are [B x T=1 x C x H x W]
            out = out.unsqueeze(1)
            # optionally apply an output nonlinearity
            out_mean = self.out_mean(out[:, :, :2, ...])  # mean predictions
            out_std = self.out_var(out[:, :, 2:, ...])  # var predictions > 0
            out = torch.cat((out_mean, out_std), dim=2)  # stack mean and var predictions

            if return_att:
                return out, None
            if self.return_maps:
                return out, maps
            else:
                return out


class UTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[2],
        out_nonlin_mean=False,
        out_nonlin_var="relu",
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        norm_skip="batch",
        norm_up="batch",
        decoder_norm="batch",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
        positional_encoding=True,
        scale_by=1,
        cond_dim=None,
        temp_encoding="ltae",
        hidden_dim=64 # for RecUNet
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
                - none: apply no normalization
            norm_skip (str): similar to encoder_norm, just controlling the normalization after convolving skipped maps
            norm_up (str): similar to encoder_norm, just controlling the normalization after transposed convolution
            decoder_norm (str): similar to encoder_norm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        self.scale_by = scale_by
        self.temp_encoding = temp_encoding

        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        # TODO:
        # insert two parallel ResNet branches:
        #   i) for coarse GTSM, with conventional CONV kernels
        #   ii) for sparse points, with partial CONV kernels # TODO: inspect outputs, still containing NaNs?

        """
        import util.partialconv.models.partialconv2d as partialconv2d
        # see https://github.com/NVIDIA/partialconv, https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py

        self.in_conv_sparse     = partialconv2d.PartialConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True, multi_channel=False, return_mask=True)
        self.in_conv_sparse2    = partialconv2d.PartialConv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True, multi_channel=False, return_mask=True)

        self.in_conv_coarse     = nn.Conv2d(in_channels=5-1, out_channels=16, kernel_size=3, padding=1, stride=1, padding_mode="reflect")
        self.in_conv_coarse2    = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, padding_mode="reflect")
        input_dim               = 2*32
        """

        # ENCODER
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0]],
            k=1,
            s=1,
            p=0,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
            cond_dim=cond_dim,
        )

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
                cond_dim=cond_dim,
            )
            for i in range(self.n_stages - 1)
        )
        # DECODER
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm_skip=norm_skip,  # 'batch'
                norm_up=norm_up,  # 'batch'
                norm=decoder_norm,  # "batch",
                padding_mode=padding_mode,
                cond_dim=cond_dim,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        # LTAE
        if self.temp_encoding=='ltae':
            self.temporal_encoder = LTAE2d(
                in_channels=encoder_widths[-1],
                d_model=d_model,
                n_head=n_head,
                mlp=[d_model, encoder_widths[-1]],
                return_att=True,
                d_k=d_k,
                positional_encoding=positional_encoding,
            )
        # RecUNet with LSTM
        elif self.temp_encoding=='lstm':
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
            agg_mode='mean'

        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        # note: not including normalization layer and ReLU nonlinearity into the final ConvBlock
        #       if inserting >1 layers into out_conv then consider treating normalizations separately
        self.out_dims = out_conv[-1]

        # spatial parameter sharing: MLP, i.e. 1 x 1 CONV
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv,
            k=1,
            s=1,
            p=0,
            padding_mode=padding_mode,
            norm="none",
            last_relu=False,
        )
        if out_nonlin_mean:
            self.out_mean = lambda vars: self.scale_by * nn.Sigmoid()(
                vars
            )  # this is for predicting mean values in [0, 1]
        else:
            self.out_mean = lambda vars: nn.Identity()(
                vars
            )  # just keep the mean estimates, without applying a nonlinearity

        if out_nonlin_var == "relu":
            self.out_var = nn.ReLU()  # this is for predicting var values > 0
        elif out_nonlin_var == "softplus":
            self.out_var = nn.Softplus(
                beta=1, threshold=20
            )  # a smooth approximation to the ReLU function
        elif out_nonlin_var == "elu":
            self.out_var = lambda vars: nn.ELU()(vars) + 1 + 1e-8
        else:  # just keep the variance estimates,
            self.out_var = (
                nn.Identity()
            )  # just keep the variance estimates, without applying a nonlinearity

    def forward(self, input, batch_positions=None, lead=None, return_att=False):
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT pad mask
        # SPATIAL ENCODER
        # collect feature maps in list 'feature_maps'

        """
        db = 42
        # process sparse input channels
        b, t, c, h, w = input.shape
        input_sparse, mask = self.in_conv_sparse(input[:,[0], ...].view(b*t, 1, h, w), torch.isnan(input[:,[0], ...].view(b*t, 1, h, w)).float())
        input_sparse, mask = self.in_conv_sparse2(input_sparse, mask)

        input_coarse = self.in_conv_coarse(input[:,1:, ...].reshape(b*t, 5-1, h, w))
        input_coarse = self.in_conv_coarse2(input_coarse)
        # x and mask are [b*t, C_out, h, w] and [b*t, 1, h, w], respectively
        # process non-sparse input channels

        input = torch.cat([input_sparse.view(b, t, input_sparse.shape[1], h, w), input_coarse.view(b, t, input_coarse.shape[1], h, w)], dim=2)
        """

        out = self.in_conv.smart_forward(input, lead)
        feature_maps = [out]
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1], lead)
            feature_maps.append(out)
        # TEMPORAL ENCODER

        if self.temp_encoding == 'ltae':
            out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
            )
        elif self.temp_encoding == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        

        # feature_maps[-1].shape is torch.Size([B, T, 128, 32, 32])
        #   -> every attention pixel has an 8x8 receptive field
        # att.shape is torch.Size([h, B, T, 32, 32])
        # out.shape is torch.Size([B, 128, 32, 32]), in self-attention class it's Size([B*32*32*h=32768, 1, 16]
        
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        
        if self.temp_encoding == 'lstm':
            att = None
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip, lead)
            if self.return_maps:
                maps.append(out)


        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            # append a singelton temporal dimension such that outputs are [B x T=1 x C x H x W]
            out = out.unsqueeze(1)
            # optionally apply an output nonlinearity
            out_mean = self.out_mean(out[:, :, :2, ...])  # mean predictions
            out_std = self.out_var(out[:, :, 2:, ...])  # var predictions > 0
            out = torch.cat((out_mean, out_std), dim=2)  # stack mean and var predictions

            if return_att:
                return out, att
            if self.return_maps:
                return out, maps
            else:
                return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input, lead=None):
        if len(input.shape) == 4:
            return self.forward(input, lead)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w), lead).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(self.out_shape, device=input.device, requires_grad=False)
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask], lead)
                    out = temp
                else:
                    out = self.forward(out, lead)
            else:
                out = self.forward(out, lead)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
        cond_dim=None,  # lead_time_embed_dim
    ):
        super(ConvLayer, self).__init__()

        self.mlp = None  # TODO: FiLM

        layers = []
        out_dim = []
        self.norm_then_relu = []

        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            out_dim.append(nkernels[i + 1])
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            uses_relu = False
            if last_relu:  # append a ReLU after the current CONV layer
                layers.append(nn.ReLU())
                uses_relu = True
            elif i < len(nkernels) - 2:  # only append ReLU if not last layer
                layers.append(nn.ReLU())
                uses_relu = True
            self.norm_then_relu.append(exists(nl) and uses_relu)
        self.conv = nn.Sequential(*layers)

        # TODO: FiLM
        # for each Conv2d followed by normalization, define a separate MLP + NOT followed by ReLU
        # note: for Met-Net3, 1 block is always 2 CONV in https://github.com/lucidrains/metnet3-pytorch/blob/a0b107d7f4b792f612341a33b63b54f79200f998/metnet3_pytorch/metnet3_pytorch.py#L164

        if exists(cond_dim):
            self.mlp = [
                nn.Sequential(
                    nn.ReLU(),
                    # linear mapping from embedding dimension to channel dimensions of CONV
                    nn.Linear(cond_dim, dim_out * 2),
                ).to("cuda")
                for idx, dim_out in enumerate(out_dim)
                if self.norm_then_relu[idx]
            ]

    def forward(self, input, lead=None):
        # TODO: FiLM

        assert not (exists(self.mlp) ^ exists(lead))

        # at each MLP (1 MLP per normalized CONV layer) compute the conditioning
        scale_shift = None
        if exists(self.mlp) and exists(lead):
            lead = [layer(lead) for layer in self.mlp]
            lead = [
                rearrange(lead_item.flatten(start_dim=1), "b c -> b c 1 1") for lead_item in lead
            ]
            scale_shift = [lead_item.chunk(2, dim=1) for lead_item in lead]

        idx = 0
        for _, layer in enumerate(self.conv):
            input = layer(input)
            isNormLayer = np.any(
                [
                    isinstance(layer, norm)
                    for norm in [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm]
                ]
            )
            # apply linear transform on CONV output
            # (following a normalization, and preceding an output non-linearity)
            if (
                exists(scale_shift)
                and isNormLayer
                and len(self.norm_then_relu) > idx
                and self.norm_then_relu[idx]
            ):
                scale, shift = scale_shift[idx]
                replicate_each_t = int(input.shape[0] / shift.shape[0])
                input = input * (scale.repeat(replicate_each_t, 1, 1, 1) + 1) + shift.repeat(
                    replicate_each_t, 1, 1, 1
                )
                idx = idx + 1
        return input  # self.conv(input, lead)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        k=3,
        s=1,
        p=1,
        padding_mode="reflect",
        cond_dim=None,  # lead_time_embed_dim
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
            cond_dim=cond_dim,
        )

    def forward(self, input, lead=None):
        return self.conv(input, lead)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
        cond_dim=None,  # lead_time_embed_dim
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
            cond_dim=cond_dim,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode, cond_dim=cond_dim
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
            cond_dim=None,  # note: even if using FiLM, don't apply on this layer
            last_relu=False,  # note: removing last ReLU in DownConvBlock because it adds onto residual connection
        )

    def forward(self, input, lead=None):
        out = self.down(input, lead)
        out = self.conv1(out, lead)
        out = out + self.conv2(out)
        return out


def get_norm_layer(out_channels, num_feats, n_groups=4, layer_type="BatchNorm"):
    if layer_type == "batch":
        return nn.BatchNorm2d(out_channels)
    elif layer_type == "instance":
        return nn.InstanceNorm2d(out_channels)
    elif layer_type == "group":
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        norm_skip="batch",
        norm_up="batch",
        norm="batch",
        n_groups=4,
        d_skip=None,
        padding_mode="reflect",
        cond_dim=None,  # lead_time_embed_dim
    ):
        super(UpConvBlock, self).__init__()

        d = d_out if d_skip is None else d_skip
        out_dims = []
        # self.norm_skip, self.norm_up= norm_skip, norm_up

        # apply another CONV and norm to the skipped paired map
        """"
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        """
        if norm_skip in ["group", "batch", "instance"]:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
                get_norm_layer(d, d, n_groups, norm_skip),  # nn.BatchNorm2d(d),
                # TODO: FiLM here
                nn.ReLU(),
            )
            out_dims.append(d)
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1), nn.ReLU()
            )

        # transposed CONV layer to perform upsampling
        """
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        """
        if norm_up in ["group", "batch", "instance"]:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
                ),
                get_norm_layer(d_out, d_out, n_groups, norm_up),  # nn.BatchNorm2d(d_out),
                # TODO: FiLM here
                nn.ReLU(),
            )
            out_dims.append(d_out)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
                ),
                nn.ReLU(),
            )

        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out],
            norm=norm,
            cond_dim=cond_dim,  # TODO: FiLM here
            padding_mode=padding_mode,  # removing  downsampling relu in UpConvBlock because of MobileNet2
        )
        out_dims.append(d_out)
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
            last_relu=False,  # removing  last relu in UpConvBlock because it adds onto residual connection
        )

        if exists(cond_dim):
            # normalization layers of: skip connection CONV, upsampling CONV and CONV1
            num_mlp = [
                norm_skip in ["group", "batch", "instance"],
                norm_up in ["group", "batch", "instance"],
                True,
            ]
            # define an MLP for each of these
            self.mlp = [
                nn.Sequential(
                    nn.ReLU(),
                    # linear mapping from embedding dimension to channel dimensions of CONV
                    nn.Linear(cond_dim, dim_out * 2),
                ).to("cuda")
                for idx, dim_out in enumerate(out_dims)
                if num_mlp[idx]
            ]
            self.out_dims = out_dims
        else:
            self.mlp = None

    def forward(self, input, skip, lead=None):

        assert not (exists(self.mlp) ^ exists(lead))

        # at each MLP (1 MLP per normalized CONV layer) compute the conditioning
        scale_shift = None
        if exists(self.mlp) and exists(lead):
            lead_time = [layer(lead) for layer in self.mlp]
            lead_time = [
                rearrange(lead_item.flatten(start_dim=1), "b c -> b c 1 1")
                for lead_item in lead_time
            ]
            scale_shift = [lead_item.chunk(2, dim=1) for lead_item in lead_time]

        idx = 0
        for idx, layer in enumerate(self.skip_conv):
            skip = layer(skip)
            isNormLayer = np.any(
                [
                    isinstance(layer, norm)
                    for norm in [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm]
                ]
            )
            # apply linear transform on CONV output
            # (following a normalization, and preceding an output non-linearity)
            if exists(scale_shift) and isNormLayer:
                scale, shift = scale_shift[idx]
                skip = skip * (scale + 1) + shift
                # replicate_each_t = int(input.shape[0]/shift.shape[0])
                # skip = skip * (scale.repeat(replicate_each_t,1,1,1) + 1) + shift.repeat(replicate_each_t,1,1,1)
                idx = idx + 1

        for idx, layer in enumerate(self.up):
            input = layer(input)
            isNormLayer = np.any(
                [
                    isinstance(layer, norm)
                    for norm in [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm]
                ]
            )
            # apply linear transform on CONV output
            # (following a normalization, and preceding an output non-linearity)
            if (
                exists(scale_shift) and isNormLayer and isinstance(self.up[-1], nn.ReLU)
            ):  # and self.norm_then_relu[idx]:
                scale, shift = scale_shift[idx]
                input = input * (scale + 1) + shift
                # replicate_each_t = int(input.shape[0]/shift.shape[0])
                # input = input * (scale.repeat(replicate_each_t,1,1,1) + 1) + shift.repeat(replicate_each_t,1,1,1)
                idx = idx + 1
        out = input

        # out = self.up(input, lead) # transposed CONV on previous decoder layer
        # apply another CONV and norm to the skipped input               --> paired encoder map
        # skip_conv = self.skip_conv(skip)
        out = torch.cat([out, skip], dim=1)  # concat '' with paired encoder map
        out = self.conv1(out, lead)  # CONV again
        out = out + self.conv2(out)  # conv with residual
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class RecUNet(nn.Module):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[13],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
        )

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="mean")

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv, k=1, s=1, p=0, padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None):
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask)
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if self.return_maps:
                return out, maps
            else:
                return out
