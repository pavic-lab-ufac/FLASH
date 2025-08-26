# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import functional as F


def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, device=None):
        super(VGGPerceptualLoss, self).__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        blocks = [
            torchvision.models.vgg16(weights="DEFAULT").features[:4].eval(),
            torchvision.models.vgg16(weights="DEFAULT").features[4:9].eval(),
            torchvision.models.vgg16(weights="DEFAULT").features[9:16].eval(),
            torchvision.models.vgg16(weights="DEFAULT").features[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.device = device
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, device=None):
        super(EdgeLoss, self).__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).to(device)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x, y):
        return torch.nn.functional.l1_loss(
            self.laplacian_kernel(x), self.laplacian_kernel(y)
        )


class FFTLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(FFTLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return torch.nn.functional.l1_loss(pred_fft, target_fft)


class Ternary(nn.Module):
    def __init__(self, patch_size=7, device=None):
        super(Ternary, self).__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        w = np.transpose(w, (3, 2, 0, 1))
        self.w = torch.tensor(w).float().to(device)

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2)
        loc_diff = patches - tensor_
        return loc_diff / torch.sqrt(0.81 + loc_diff**2)

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        return F.pad(inner, [padding] * 4)

    def forward(self, x, y):
        diff = self.transform(x) - self.transform(y).detach()
        dist = (diff**2 / (0.1 + diff**2)).mean(dim=1, keepdim=True)
        return (dist * self.valid_mask(x)).mean()


class tanh_L1Loss(nn.Module):
    def forward(self, x, y):
        return torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))


class tanh_L2Loss(nn.Module):
    def forward(self, x, y):
        return torch.mean((torch.tanh(x) - torch.tanh(y)) ** 2)


# class CustomLoss(nn.Module):
#     def __init__(self, alpha=0.01, beta=0.01, device=None):
#         super(CustomLoss, self).__init__()
#         device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.alpha = alpha
#         self.beta = beta
#         self.loss_tl1 = tanh_L1Loss()
#         # self.edge = Ternary(device=device)
#         self.lpips = VGGPerceptualLoss(device=device)
#         self.device = device
#         self.census = Ternary(device=device)

#     def forward(self, input, target):
#         input = range_compressor(input)
#         target = range_compressor(target)

#         l1 = torch.nn.functional.l1_loss(input, target)
#         lp = self.lpips(input, target)
#         # le = self.edge(input, target)
#         lc = self.census(input,target)

#         return l1 + self.alpha * lp + 0.1 * (l1 + lc)
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.01, device=None):
        super(CustomLoss, self).__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.loss_tl1 = tanh_L1Loss()
        # self.edge = Ternary(device=device)
        self.lpips = VGGPerceptualLoss(device=device)
        self.device = device
        self.census = Ternary(device=device)

    def forward(self, input, target):
        # input = range_compressor(input)
        # target = range_compressor(target)

        l1 = torch.nn.functional.l1_loss(input, target)
        lp = self.lpips(input, target)
        # le = self.edge(input, target)
        lc = self.census(input,target)

        return l1 + self.alpha * lp + 0.1 * (l1 + lc)
    

class KnowledgeLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.01, device=None):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.loss_tl1 = tanh_L1Loss()
        # self.edge = Ternary(device=device)
        self.lpips = VGGPerceptualLoss(device=device)
        self.device = device
        self.census = Ternary(device=device)

    def forward(self, input, target, teacher):
        input = range_compressor(input)
        target = range_compressor(target)
        teacher = range_compressor(teacher)

        l1 = torch.nn.functional.l1_loss(input, target)
        lp = self.lpips(input, teacher)
        # le = self.edge(input, target)
        # lc = self.census(input,teacher)

        return l1 + self.alpha * lp



def gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.unsqueeze(0).unsqueeze(0)  # shape (1,1,window_size,window_size)


def ssim_torch(img1, img2, window_size=11, sigma=1.5, data_range=255):
    
    r, g, b = img1[:, 0:1, :, :], img1[:, 1:2, :, :], img1[:, 2:3, :, :]
    # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img1 = 1/3 * r + 1/3 * g + 1/3 * b

    r, g, b = img2[:, 0:1, :, :], img2[:, 1:2, :, :], img2[:, 2:3, :, :]
    # img2 = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img2 = 1/3 * r + 1/3 * g + 1/3 * b
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    device = img1.device
    window = gaussian_window(window_size, sigma, device)

    # Add channel and batch dims if missing
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=0)
    mu2 = F.conv2d(img2, window, padding=0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=0) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def psnr_torch(img1, img2, data_range=1):
    return 10 * torch.log10(data_range / ((img1 - img2).mean() + 1e-8))    

class ValLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        
        psnr = psnr_torch(input, target)
        ssim = ssim_torch(input*255, target*255)
        
        input = range_compressor(input)
        target = range_compressor(target)

        psnr_u = psnr_torch(input, target)
        ssim_u = ssim_torch(input*255, target*255)

        return psnr_u, ssim_u, psnr, ssim


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    customloss = CustomLoss(device=device)
    valloss = ValLoss()

    x1 = torch.ones((32, 3, 128, 128)).to(device)
    x2 = torch.ones((32, 3, 128, 128)).to(device)
    x3 = torch.ones((32, 3, 128, 128)).to(device)

    res = customloss(x1, x2)
    print(f"Loss computed: {res.item()}")
