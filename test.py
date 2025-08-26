# -*- coding:utf-8 -*-

import os.path as osp
import argparse
import time
import os
import shutil

# from dataset.dataset_iccv23 import ICCV23_Test_Dataset
from dataset.dataset_sig17 import (
    SIG17_Training_Dataset,
    SIG17_Validation_Dataset,
    SynHDR_Test_Dataset,
    Real_Test_Dataset,
    Tursun_Test_Dataset)
from torch.utils.data import DataLoader
from utils.utils import *
import importlib.util
import sys

# from pytorch_msssim import ssim

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument(
    "--dataset_dir", type=str, default="./our_data", help="dataset directory"
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=1,
    metavar="N",
    help="testing batch size (default: 1)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    metavar="N",
    help="number of workers to fetch data (default: 1)",
)
parser.add_argument("--patch_size", type=int, default=2000)
parser.add_argument("--ckpt", type=str, default="./ckpt_sctnet/")
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--isTLC", default=False, action="store_true")
parser.add_argument("--loader", default="Kalantari", choices=["Kalantari", "SynHDR", "Tursun", "Real"])
times = []


def main():
    # Settings
    args = parser.parse_args()

    if not args.save_dir:
        args.save_dir = os.path.splitext(args.ckpt)[0]
    mod = import_from_path("FLASH", os.path.dirname(args.ckpt) + "/FLASH.py")
    FLASH = mod.FLASH

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>", flush=True)
    print("Load weights from: ", args.ckpt, flush=True)
    print(f"Is TLC: {args.isTLC}")
    
    args.save_dir = args.save_dir if not args.isTLC else f"{args.save_dir}_TLC"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print(device, flush=True)

    model = FLASH().to(device)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(
        torch.load(
            f"{args.ckpt}", map_location=torch.device(device), weights_only=True
        )["state_dict"]
    )

    model.eval()

    if args.loader == "Kalantari":
        datasets = SIG17_Validation_Dataset(args.dataset_dir, crop=False)
    elif args.loader == "SynHDR":
        datasets = SynHDR_Test_Dataset(args.dataset_dir)
    elif args.loader == "Tursun":
        datasets = Tursun_Test_Dataset(args.dataset_dir)
    elif args.loader == "Real":
        datasets = Real_Test_Dataset(args.dataset_dir)
        
        
    dataloader = DataLoader(
        dataset=datasets, batch_size=1, num_workers=1, shuffle=False
    )
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()

    gpu_memory_list = []

    if osp.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "metrics.out"), "+w") as f:

        f.write(">>>>>>>>> Start Testing >>>>>>>>>\n")
        f.write(f"Load weights from: {args.ckpt}\n")
        f.write(f"Is TLC: {args.isTLC}\n")
        f.write(f"{device}\n")
        f.write("\n")
        None

    for i, images in enumerate(dataloader):

        # Comenzar a medir tiempo
        img0_c = images["input0"].to(device)
        img1_c = images["input1"].to(device)
        img2_c = images["input2"].to(device)
        label = images["label"].to(device)
        start_time = time.time()
        with torch.no_grad():
            pred_img = model(img0_c, img1_c, img2_c, args.isTLC)
        end_time = time.time()
        pred_hdr = pred_img
        
        label_mu = range_compressor_tensor(label, device)
        pred_img_mu = range_compressor_tensor(pred_img, device)
        
        scene_psnr_l = calculate_psnr_cuda(label, pred_img)
        scene_psnr_mu = calculate_psnr_cuda(label_mu, pred_img_mu)
        
        scene_ssim_l = ssim_matlab(pred_img, label)
        scene_ssim_mu = ssim_matlab(pred_img_mu, label_mu)
        
        
        pred_img = pred_img[0].cpu().permute(1, 2, 0).numpy()
        label = label[0].cpu().permute(1, 2, 0).numpy()
        pred_img_mu = pred_img_mu[0].cpu().permute(1, 2, 0).numpy()
        label_mu = label_mu[0].cpu().permute(1, 2, 0).numpy()
        

        pred_img = np.clip(pred_img * 255.0, 0.0, 255.0)
        label = np.clip(label * 255.0, 0.0, 255.0)
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0.0, 255.0)
        label_mu = np.clip(label_mu * 255.0, 0.0, 255.0)
        

        logout = f" {i} | PSNR_mu: {scene_psnr_mu:.4f}  PSNR_l: {scene_psnr_l:.4f} | SSIM_mu: {scene_ssim_mu:.4f}  SSIM_l: {scene_ssim_l:.4f} Seconds: {end_time - start_time:.4f}"

        print(logout, flush=True)
        with open(os.path.join(args.save_dir, "metrics.out"), "a") as f:
            f.write(logout)
            f.write("\n")

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        if i != 0:
            times.append(end_time - start_time)

        # save results
        # args.save_dir = f"{args.ckpt[:-4]}"

        pred_hdr = pred_hdr[0].cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(args.save_dir, "{}_pred.png".format(i)), pred_img_mu)
        cv2.imwrite(os.path.join(args.save_dir, "{}_pred.hdr".format(i)), pred_hdr)
        cv2.imwrite(os.path.join(args.save_dir, "{}_gt.png".format(i)), label_mu)

    print(
        "Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg),
        flush=True,
    )
    print(
        "Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg),
        flush=True,
    )
    print(f"Average time {np.mean(times)}", flush=True)
    print(">>>>>>>>> Finish Testing >>>>>>>>>", flush=True)

    with open(os.path.join(args.save_dir, "metrics.out"), "a") as f:
        f.write(
            "Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg)
        )
        f.write("\n")
        f.write(
            "Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg)
        )
        f.write("\n")
        f.write(f"Average time {np.mean(times)}")
        f.write("\n")
        f.write(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == "__main__":
    main()
