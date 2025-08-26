# -*- coding:utf-8 -*-
import numpy as np
import os, glob
import cv2
import math
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init

# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio
import math
import numpy as np
import torch
import torch.nn.functional as F
from math import exp

# imageio.plugins.freeimage.download()
os.environ["IMAGEIO_FREEIMAGE_PATH"] = (
    "/home/urso/SCTNet/utils/libfreeimage-3.16.0-linux64.so"
)


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        1,
    )
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
    sum_img = torch.zeros_like(ldr_imgs[1])
    sum_w = torch.zeros_like(ldr_imgs[1])
    w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
    w_mid = (
        weight_3expo_mid_tog17(ldr_imgs[1])
        + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0)
        + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
    )
    w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
    w_list = [w_low, w_mid, w_high]
    for i in range(len(ldr_imgs)):
        sum_w += w_list[i]
        sum_img += w_list[i] * lin_imgs[i]
    hdr_img = sum_img / (sum_w + 1e-9)
    return hdr_img


def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w


def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w


def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w


def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))


def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))


def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)
        # equivalent to im2single from Matlab
        img = img / 2**16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)


def read_images_custom(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, 1)
        
        # h, w = img.shape[:2]
        # if h < w:
        #     scale = 1500.0 / h
        # else:
        #     scale = 1500.0 / w
        # new_w = int(w * scale)
        # new_h = int(h * scale)
        # img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # equivalent to im2single from Matlab    
        maxx = np.max(img)    
        if maxx <= 1:
            img = img / 1.0
        elif maxx <= 255.0:
            img = img / 255.0
        elif maxx <= 65535.0:
            img = img / 65535.0
        else:
            img = img / maxx
        img = np.float32(img)
        
        # c1 = img[:,:,0]
        # c2 = img[:,:,0]
        # c3 = img[:,:,0]
        
        
        
        img = np.clip(img, 0, np.percentile(img, 99))
        imgs.append(img)
    return np.array(imgs)



def read_label(file_path, file_name, alternative_name=""):
    try:
        label = imageio.imread(os.path.join(file_path, file_name), "hdr")
    except:
        label = imageio.imread(os.path.join(file_path, alternative_name), "hdr")
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label


def ldr_to_hdr(imgs, expo, gamma):
    return (imgs**gamma) / (expo + 1e-8)


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1 / sqrdErr)


def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(
            Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range
        )
    return psnr / Img.shape[0]


def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(
            Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range
        )
    return psnr / Img.shape[0]


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_parameters(net, scale=0.1):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            m.weight.data *= scale
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:

        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            for i in range(3):
                res = []
                res.append(ssim(img1[:,:,i], img2[:,:,i]))
                return np.mean(res)
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    return radiance_writer(path, image)


def calculate_psnr_cuda(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf')).to(img1.device)
    psnr = 10 * torch.log10((1 ** 2) / mse)
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1, device="cpu"):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = (
        _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    )
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def create_window_3d(window_size, channel=1, device="cpu"):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = (
        _3D_window.expand(1, channel, window_size, window_size, window_size)
        .contiguous()
        .to(device)
    )
    return window


def ssim_matlab(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=1,
):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(
        F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"),
        window,
        padding=padd,
        groups=1,
    )
    mu2 = F.conv3d(
        F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"),
        window,
        padding=padd,
        groups=1,
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(
            F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), "replicate"),
            window,
            padding=padd,
            groups=1,
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(
            F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), "replicate"),
            window,
            padding=padd,
            groups=1,
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv3d(
            F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), "replicate"),
            window,
            padding=padd,
            groups=1,
        )
        - mu1_mu2
    )

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret