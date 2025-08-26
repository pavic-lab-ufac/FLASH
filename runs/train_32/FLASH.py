import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_safnet import warp, merge_hdr
from mamba_ssm import Mamba
import numpy as np
from torchvision.ops import DeformConv2d


div_size = 16
div_flow = 20.0


def window_partition(x, window_size):
    """
    Divide la imagen en ventanas (parches) de tamaño fijo.

    Args:
        x: (B, C, H, W) - Tensor de entrada con canales primero.
        window_size (int): Tamaño de la ventana (parche).

    Returns:
        windows: (num_windows*B, C, window_size, window_size) - Ventanas reorganizadas.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = (
        x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reconstruye la imagen original a partir de las ventanas (parches).

    Args:
        windows: (num_windows*B, C, window_size, window_size) - Ventanas reorganizadas.
        window_size (int): Tamaño de la ventana (parche).
        H (int): Altura de la imagen original.
        W (int): Ancho de la imagen original.

    Returns:
        x: (B, C, H, W) - Imagen reconstruida.
    """
    B = int(
        windows.shape[0] / ((H * W) / (window_size * window_size))
    )  # Calcula el batch size original
    C = windows.shape[1]  # Número de canales
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x


def resize(x, scale_factor):
    return F.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=True,
    )


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.PReLU(out_channels),
    )


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=True
    )


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(6, 40, 3, 2, 1), convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), convrelu(40, 40, 3, 1, 1)
        )

    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=bias,
            ),
            nn.PReLU(channels),
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class VisionMambaEncoder2D(nn.Module):
    def __init__(self, channels, ssm_config=None):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

        # Proyecciones lineales
        self.to_x = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_z = nn.Conv2d(channels, channels, kernel_size=1)

        # Horizontal convoluciones
        self.forward_conv_h = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels
        )
        self.backward_conv_h = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels
        )

        # Vertical convoluciones
        self.forward_conv_v = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )
        self.backward_conv_v = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )

        # SSMs (uno por dirección)
        self.forward_ssm_h = Mamba(d_model=channels, **(ssm_config or {}))
        self.backward_ssm_h = Mamba(d_model=channels, **(ssm_config or {}))
        self.forward_ssm_v = Mamba(d_model=channels, **(ssm_config or {}))
        self.backward_ssm_v = Mamba(d_model=channels, **(ssm_config or {}))

        self.activation = nn.GELU()
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # LayerNorm por canal
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm(x_perm).permute(0, 3, 1, 2)  # [B, C, H, W]

        x_proj = self.to_x(x_norm)
        z_proj = self.to_z(x_norm)
        z_activated = self.activation(z_proj)

        # === Horizontal ===
        # Forward
        h_fwd_h = self.forward_conv_h(x_proj)
        h_fwd_h = h_fwd_h.flatten(2).transpose(1, 2)  # [B, H*W, C]
        h_fwd_h = self.forward_ssm_h(h_fwd_h).transpose(1, 2).reshape(B, C, H, W)

        # Backward
        x_flip_w = torch.flip(x_proj, dims=[3])
        h_bwd_h = self.backward_conv_h(x_flip_w)
        h_bwd_h = h_bwd_h.flatten(2).transpose(1, 2)
        h_bwd_h = self.backward_ssm_h(h_bwd_h).transpose(1, 2).reshape(B, C, H, W)
        h_bwd_h = torch.flip(h_bwd_h, dims=[3])

        # === Vertical ===
        # Forward
        h_fwd_v = self.forward_conv_v(x_proj)
        h_fwd_v = h_fwd_v.permute(0, 3, 2, 1).reshape(B * W, H, C)  # eje H
        h_fwd_v = self.forward_ssm_v(h_fwd_v).reshape(B, W, H, C).permute(0, 3, 2, 1)

        # Backward
        x_flip_h = torch.flip(x_proj, dims=[2])
        h_bwd_v = self.backward_conv_v(x_flip_h)
        h_bwd_v = h_bwd_v.permute(0, 3, 2, 1).reshape(B * W, H, C)
        h_bwd_v = self.backward_ssm_v(h_bwd_v).reshape(B, W, H, C).permute(0, 3, 2, 1)
        h_bwd_v = torch.flip(h_bwd_v, dims=[2])

        # Combinación
        y = (h_fwd_h + h_bwd_h + h_fwd_v + h_bwd_v) * z_activated

        # Proyección y conexión residual
        y = self.output_proj(y)
        return torch.clamp(y + x, -100, 100)


class SFTLayer(nn.Module):
    def __init__(self, x0_ch, x1_ch):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(x1_ch, x0_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(x0_ch, x0_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(x1_ch, x0_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(x0_ch, x0_ch, 1)

    def forward(self, x0, x1):
        """Caracteristicas de x0 condicionadas con x1"""
        scale = self.SFT_scale_conv1(
            F.leaky_relu(self.SFT_scale_conv0(x1), 0.1, inplace=True)
        )
        shift = self.SFT_shift_conv1(
            F.leaky_relu(self.SFT_shift_conv0(x1), 0.1, inplace=True)
        )
        return x0 * (scale + 1) + shift


class DCRB(nn.Module):
    def __init__(self, x0_ch, x1_ch):
        super(DCRB, self).__init__()
        dch = 2 * 3 * 3
        self.offset = nn.Conv2d(x0_ch, dch, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(x0_ch, x1_ch)
        self.dconv1 = DeformConv2d(x0_ch, x0_ch, 3, 1, 1)  # Todo :DCONV
        self.sft2 = SFTLayer(x0_ch, x1_ch)
        self.dconv2 = DeformConv2d(x0_ch, x0_ch, 3, 1, 1)  # Todo :DCONV

    def forward(self, x0, x1):
        # x[0]: fea; x[1]: cond
        off = self.offset(x0)
        fea = self.sft1(x0, x1)
        fea = F.relu(self.dconv1(fea, off), inplace=True)
        fea = self.sft2(fea, x1)
        fea = self.dconv2(fea, off)
        return x0 + fea


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv0 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))
        self.conv1 = nn.Sequential(
            convrelu(6 + 2 + 2 + 1 + 1 + 3, 40), convrelu(40, 40)
        )
        self.conv2 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))

        self.mambablocks = nn.ModuleList()
        self.lowAlign = nn.ModuleList()
        self.highAlign = nn.ModuleList()
        self.conBlock = nn.ModuleList()

        dim_f = 20
        dim_c = 40
        layers = 2
        for _ in range(layers):
            mam = nn.Sequential(
                VisionMambaEncoder2D(dim_c),
                VisionMambaEncoder2D(dim_c),
            )
            low = DCRB(dim_f, dim_c)
            high = DCRB(dim_f, dim_c)
            con = nn.Sequential(
                nn.Conv2d(dim_c + dim_f + dim_f, dim_c, 3, 1, 1),
                nn.PReLU(dim_c),
                ResBlock(dim_c),
            )

            self.mambablocks.append(mam)
            self.lowAlign.append(low)
            self.highAlign.append(high)
            self.conBlock.append(con)

        self.conv_end = nn.Sequential(nn.Conv2d(dim_c, 3, 3, 1, 1))

    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        # window_size = 128
        # _, _, H, W = img_hdr_m.shape
        # img0_c = window_partition(img0_c, window_size)
        # img1_c = window_partition(img1_c, window_size)
        # img2_c = window_partition(img2_c, window_size)
        # flow0 = window_partition(flow0, window_size)
        # flow2 = window_partition(flow2, window_size)
        # mask0 = window_partition(mask0, window_size)
        # mask2 = window_partition(mask2, window_size)
        # img_hdr_m = window_partition(img_hdr_m, window_size)

        feat0 = self.conv0(img0_c)
        feat0 = warp(feat0, flow0)
        feat1 = self.conv1(
            torch.cat(
                [img1_c, flow0 / div_flow, flow2 / div_flow, mask0, mask2, img_hdr_m], 1
            )
        )
        feat2 = self.conv2(img2_c)
        feat2 = warp(feat2, flow2)
        
        


        for mambablock, low, high, conv in zip(
            self.mambablocks,
            self.lowAlign,
            self.highAlign,
            self.conBlock,
        ):
            feat0 = low(feat0, feat1)
            feat2 = high(feat2, feat1)
            feat = torch.cat((feat0, feat1, feat2), 1)
            feat = conv(feat)
            feat1 = feat * torch.sigmoid(mambablock(feat))
            
            
                
        res = self.conv_end(feat1)

        img_hdr_r = torch.clamp(img_hdr_m + res, 1e-8, 1)
        # img_hdr_r = window_reverse(img_hdr_r, window_size, H, W)

        return img_hdr_r


class FLASH(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size)),
        )

        if input_size != org_size:
            img0_c = F.interpolate(
                img0_c, size=input_size, mode="bilinear", align_corners=False
            )
            img1_c = F.interpolate(
                img1_c, size=input_size, mode="bilinear", align_corners=False
            )
            img2_c = F.interpolate(
                img2_c, size=input_size, mode="bilinear", align_corners=False
            )

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(
            f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5
        )
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(
            f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4
        )
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(
            f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3
        )
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(
            f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2
        )

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(
                up_flow0_1, size=org_size, mode="bilinear", align_corners=False
            )
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(
                up_flow2_1, size=org_size, mode="bilinear", align_corners=False
            )
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(
                up_mask0_1, size=org_size, mode="bilinear", align_corners=False
            )
            up_mask2_1 = F.interpolate(
                up_mask2_1, size=org_size, mode="bilinear", align_corners=False
            )

        up_mask0_1 = torch.sigmoid(up_mask0_1)
        up_mask2_1 = torch.sigmoid(up_mask2_1)

        return up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1

    def forward(self, img0_c, img1_c, img2_c, refine=True, scale_factor=0.5):
        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        flow0, flow2, mask0, mask2 = self.forward_flow_mask(
            img0_c, img1_c, img2_c, scale_factor=scale_factor
        )

        img0_c_warp = warp(img0_c, flow0)
        img2_c_warp = warp(img2_c, flow2)
        img_hdr_m = merge_hdr(
            [
                img0_c_warp[:, 3:6, :, :],
                img1_c[:, 3:6, :, :],
                img2_c_warp[:, 3:6, :, :],
            ],
            [
                img0_c_warp[:, 0:3, :, :],
                img1_c[:, 0:3, :, :],
                img2_c_warp[:, 0:3, :, :],
            ],
            mask0,
            mask2,
        )


        
        img_hdr_r = self.refinenet(
            img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m
        )
        return img_hdr_r


if __name__ == "__main__":

    import time
    from thop import profile

    model = FLASH().cuda().eval()

    w = 1500
    h = 1000
    # w = 128
    # h = 128
    img0_c = torch.randn(1, 6, h, w).cuda()
    img1_c = torch.randn(1, 6, h, w).cuda()
    img2_c = torch.randn(1, 6, h, w).cuda()

    flops, params = profile(model, inputs=(img0_c, img1_c, img2_c), verbose=False)
    print(
        "FLOPs: {:.3f}T of 0.976T\nParams: {:.2f}M of 1.12M".format(
            flops / 1000 / 1000 / 1000 / 1000, params / 1000 / 1000
        )
    )
    ##SAFNET 0.776 seconds, 0.976 TFlops, 1.12 MParams 1500x1000
    ##
    """
    model = FLASH()
    x = torch.randn((1, 6, 128, 128))
    x = model(x, x, x)
    print(x.shape)"""
