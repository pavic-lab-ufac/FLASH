# -*- coding:utf-8 -*-
import os
import shutil
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import *
from models.FLASH import FLASH
from dataset.dataset_sig17 import (
    SIG17_Training_Dataset,
    SIG17_Validation_Dataset,
    SynHDR_Test_Dataset,
    Real_Test_Dataset,
    Tursun_Test_Dataset
)
from models.loss import CustomLoss, ValLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from skimage.metrics import structural_similarity as ssim_scikit
from skimage.metrics import peak_signal_noise_ratio as psnr_scikit

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def get_args():
    parser = argparse.ArgumentParser(
        description="PavicHDR", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Parameters
    parser.add_argument(
        "--dataset_dir", type=str, default="./data", help="dataset directory"
    ),
    parser.add_argument(
        "--sub_set",
        type=str,
        default="sig17_training_crop128_stride64",
        help="dataset directory",
    )
    parser.add_argument(
        "--logdir", type=str, default="./checkpoints", help="target log directory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="N",
        help="number of workers to fetch data (default: 8)",
    )
    # Training
    parser.add_argument(
        "--resume", type=str, default=None, help="load model from a .pth file"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--val_epochs", type=int, default=1, metavar="S", help="random seed (default: 443)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 443)"
    )
    parser.add_argument(
        "--init_weights", action="store_true", default=False, help="init model weights"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0002,
        metavar="LR",
        help="learning rate (default: 0.0002)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.0002,
        metavar="LR",
        help="learning rate (default: 0.0002)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        metavar="N",
        help="start epoch of training (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="training batch size (default: 16)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        metavar="N",
        help="testing batch size (default: 1)",
    )
    parser.add_argument(
        "--lr_decay_interval",
        type=int,
        default=50,
        help="decay learning rate every N epochs(default: 100)",
    )
    parser.add_argument("--patch_size", type=int, default=2000),
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    loss_meter = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        # data_time.update(time.time() - end)
        batch_ldr0, batch_ldr1, batch_ldr2 = (
            batch_data["input0"].to(device),
            batch_data["input1"].to(device),
            batch_data["input2"].to(device),
        )
        label = batch_data["label"].to(device)
        pred = model(
            batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()
        )
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print(
        "Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\tTime: {:.3f}\tlr: {:6f}".format(
            epoch,
            (batch_idx + 1) * args.batch_size,
            len(train_loader.dataset),
            100.0 * (batch_idx + 1) * args.batch_size / len(train_loader.dataset),
            loss_meter.avg,
            time.time() - end,
            optimizer.param_groups[0]["lr"],
        )
    )
    with open(os.path.join(args.logdir, "log.out"), "a") as f:
        f.write(
            "Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\tTime: {:.3f}\tlr: {:6f}".format(
                epoch,
                (batch_idx + 1) * args.batch_size,
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) * args.batch_size / len(train_loader.dataset),
                loss_meter.avg,
                time.time() - end,
                optimizer.param_groups[0]["lr"],
            )
        )
        f.write("\n")


def validation(
    args, model, device, optimizer, epoch, cur_psnr, criterion, TLC=False, **kwargs
):
    model.eval()
    test_datasets = SIG17_Validation_Dataset(args.dataset_dir, crop=False)
    dataloader = DataLoader(
        dataset=test_datasets, batch_size=args.test_batch_size, num_workers=1, shuffle=False
    )
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    loss_meter = AverageMeter()
    for idx, img_dataset in enumerate(dataloader):
        img0_c = img_dataset["input0"].to(device)
        img1_c = img_dataset["input1"].to(device)
        img2_c = img_dataset["input2"].to(device)
        label = img_dataset["label"].to(device)
        
        
        with torch.no_grad():
            pred_img = model(img0_c, img1_c, img2_c, TLC)

        loss = criterion(pred_img, label)
        
        label_mu = range_compressor_tensor(label, device)
        pred_img_mu = range_compressor_tensor(pred_img, device)
        
        scene_psnr_l = calculate_psnr_cuda(label, pred_img)
        scene_psnr_mu = calculate_psnr_cuda(label_mu, pred_img_mu)
        
        scene_ssim_l = ssim_matlab(pred_img, label)
        scene_ssim_mu = ssim_matlab(pred_img_mu, label_mu)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        loss_meter.update(loss.item())

    print(
        "==Validation==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}\tLoss: {:.6f}".format(
            psnr_l.avg, psnr_mu.avg, ssim_l.avg, ssim_mu.avg, loss_meter.avg
        )
    )
    with open(os.path.join(args.logdir, "log.out"), "a") as f:
        f.write(
            "==Validation==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}\tLoss: {:.6f}".format(
                psnr_l.avg, psnr_mu.avg, ssim_l.avg, ssim_mu.avg, loss_meter.avg
            )
        )
        f.write("\n")

    # save_model
    save_dict = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(save_dict, os.path.join(args.logdir, "val_latest_checkpoint.pth"))
    if psnr_mu.avg > cur_psnr[0]:
        torch.save(save_dict, os.path.join(args.logdir, "best_checkpoint.pth"))
        cur_psnr[0] = psnr_mu.avg
        with open(os.path.join(args.logdir, "best_checkpoint.json"), "w") as f:
            f.write("best epoch:" + str(epoch) + "\n")
            f.write(
                "Validation set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}, isTLC:{}\n".format(
                    psnr_l.avg,
                    psnr_mu.avg,
                    ssim_l.avg,
                    ssim_mu.avg,
                    TLC,
                )
            )


def main():
    # settings
    args = get_args()
    # random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model architectures
    model = FLASH()
    model.to(device).eval()   
    from thop import profile
    w = 1500
    h = 1000
    img0_c = torch.randn(1, 6, h, w).to(device)
    img1_c = torch.randn(1, 6, h, w).to(device)
    img2_c = torch.randn(1, 6, h, w).to(device)
    flops, params = profile(model, inputs=(img0_c, img1_c, img2_c), verbose=False)

    model = FLASH()
    model.to(device)
    cur_psnr = [-1.0]
    # init
    # if args.init_weights:
    init_parameters(model)
    # loss
    criterion = CustomLoss().to(device)
    criterion_val = ValLoss().to(device)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr0, betas=(0.9, 0.999), eps=1e-08
    )
    model = nn.DataParallel(model)
    with open(os.path.join(args.logdir, "log.out"), "+w") as f:
        None
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            with open(os.path.join(args.logdir, "log.out"), "a") as f:
                f.write("===> Loading checkpoint from: {}".format(args.resume))
                f.write("\n")
            checkpoint = torch.load(args.resume, weights_only=True)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint["epoch"]))
            with open(os.path.join(args.logdir, "log.out"), "a") as f:
                f.write("===> Loaded checkpoint: epoch {}".format(checkpoint["epoch"]))
                f.write("\n")
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))
            with open(os.path.join(args.logdir, "log.out"), "a") as f:
                f.write("===> No checkpoint is founded at {}.".format(args.resume))
                f.write("\n")

    # dataset and dataloader
    train_dataset = SIG17_Training_Dataset(
        root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataset = SIG17_Validation_Dataset(
        root_dir=args.dataset_dir, is_training=False, crop=True, crop_size=512
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    

    dataset_size = len(train_loader.dataset)
    print(
        f"""===> Start training 

        Log dir:         {args.logdir}
        Dataset dir:     {args.dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Lr0:             {args.lr0}
        Lrf:             {args.lrf}
        Lr_interval:     {args.lr_decay_interval}
        Training size:   {dataset_size}
        Device:          {device.type}
        FLOPs (T):       {flops / 1000 / 1000 / 1000 / 1000:.2f}
        Params (M):      {params / 1000 / 1000:.2f}
        """
    )
    with open(os.path.join(args.logdir, "log.out"), "a") as f:
        f.write(
            f"""===> Start training 

        Log dir:         {args.logdir}
        Dataset dir:     {args.dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Lr0:             {args.lr0}
        Lrf:             {args.lrf}
        Lr_interval:     {args.lr_decay_interval}
        Training size:   {dataset_size}
        Device:          {device.type}
        FLOPs (T):       {flops / 1000 / 1000 / 1000 / 1000:.2f}
        Params (M):      {params / 1000 / 1000:.2f}
        """
        )
        f.write("\n")
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.lr_decay_interval, eta_min=args.lrf)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.lr_decay_interval, eta_min=args.lrf
    )
    shutil.copy("/home/urso/PavicHDR/models/PavicHDR.py", f"{args.logdir}/PavicHDR.py")
    if args.resume:
        for epoch in range(args.epochs):
            if epoch % args.val_epochs == 0:
                    validation(args, model, device, optimizer, epoch, cur_psnr, criterion)
                    validation(args, model, device, optimizer, epoch, cur_psnr, criterion, TLC=True)
            train(args, model, device, train_loader, optimizer, epoch, criterion)
            scheduler.step()
    else:
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch, criterion)
            if epoch % args.val_epochs == 0:
                    validation(args, model, device, optimizer, epoch, cur_psnr, criterion)
                    validation(args, model, device, optimizer, epoch, cur_psnr, criterion, TLC=True)
            scheduler.step()


if __name__ == "__main__":
    main()
