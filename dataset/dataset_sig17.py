# -*- coding:utf-8 -*-
import os
import os.path as osp
import sys

sys.path.append("..")
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
from glob import glob


def read_ldr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1])
    img = (img / 2**16).clip(0, 1).astype(np.float32)
    return img


def read_hdr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1]).astype(np.float32)
    return img


def read_expos(txt_path):
    expos = np.power(2, np.loadtxt(txt_path) - min(np.loadtxt(txt_path))).astype(
        np.float32
    )
    return expos


class SIG17_Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(
                self.scenes_dir, self.scenes_list[scene], "exposure.txt"
            )
            ldr_file_path = list_all_files_sorted(
                os.path.join(self.scenes_dir, self.scenes_list[scene]), ".tif"
            )
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(
            self.image_list[index][2], "label.hdr", "HDRImg.hdr"
        )  # 'label.hdr' for cropped training data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {"input0": img0, "input1": img1, "input2": img2, "label": label}
        return sample

    def __len__(self):
        return len(self.scenes_list)


class SIG17_Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # sample dir
        self.scenes_dir = osp.join(root_dir, "Test")
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(
                self.scenes_dir, self.scenes_list[scene], "exposure.txt"
            )
            ldr_file_path = list_all_files_sorted(
                os.path.join(self.scenes_dir, self.scenes_list[scene]), ".tif"
            )
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        try:
            label = read_label(
                self.image_list[index][2], "HDRImg.hdr"
            )  # 'HDRImg.hdr' for test data
        except:
            label = read_label(
                self.image_list[index][2], "hdr_img.hdr"
            )  # 'HDRImg.hdr' for test data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        if self.crop:
            x = 0
            y = 0
            img0 = (
                pre_img0[x : x + self.crop_size, y : y + self.crop_size]
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            img1 = (
                pre_img1[x : x + self.crop_size, y : y + self.crop_size]
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            img2 = (
                pre_img2[x : x + self.crop_size, y : y + self.crop_size]
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            label = (
                label[x : x + self.crop_size, y : y + self.crop_size]
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {"input0": img0, "input1": img1, "input2": img2, "label": label}
        return sample

    def __len__(self):
        return len(self.scenes_list)

class SynHDR_Test_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # sample dir
        self.scenes_dir = osp.join(root_dir, "Test")
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        print(len(self.scenes_list))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(
                self.scenes_dir, self.scenes_list[scene], "input_exp.txt"
            )
            ldr_file_path = [os.path.join(self.scenes_dir, self.scenes_list[scene], x) for x in sorted(os.listdir(os.path.join(self.scenes_dir, self.scenes_list[scene]))) if x.startswith("input") and x.endswith(".tif")]
            
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]


    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        expoTimes = [1,4,16]
        # expoTimes = [0.85, 3.2, 10]
        # Read LDR images
        ldr_images = read_images_custom(self.image_list[index][1])
        # Read HDR label
        label = read_label(
            self.image_list[index][2], "ref_hdr_aligned_linear.hdr"
        )  # 'HDRImg.hdr' for test data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)
        
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        

        sample = {"input0": img0, "input1": img1, "input2": img2, "label": label}
        return sample

    def __len__(self):
        return len(self.scenes_list)


class Real_Test_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # sample dir
        self.scenes_dir = osp.join(root_dir, "Test")
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        print(len(self.scenes_list))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(
                self.scenes_dir, self.scenes_list[scene], "input_exp.txt"
            )
            ldr_file_path = [os.path.join(self.scenes_dir, self.scenes_list[scene], x) for x in sorted(os.listdir(os.path.join(self.scenes_dir, self.scenes_list[scene]))) if x.endswith("jpg") and not x.endswith("HDR.jpg")]
            
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]


    def __getitem__(self, index):
        # Read exposure times
        # expoTimes = read_expo_times(self.image_list[index][0])
        expoTimes = [1.5, 5, 36]
        # Read LDR images
        ldr_images = read_images_custom(self.image_list[index][1])
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)
        
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        

        sample = {"input0": img0, "input1": img1, "input2": img2, "label": img1[3:6,:,:]}
        return sample

    def __len__(self):
        return len(self.scenes_list)



class Tursun_Test_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # sample dir
        self.scenes_dir = osp.join(root_dir, "Test")
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        print(len(self.scenes_list))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(
                self.scenes_dir, self.scenes_list[scene], "input_exp.txt"
            )
            ldr_file_path = (os.path.join(self.scenes_dir, self.scenes_list[scene], "3.tiff"),
                             os.path.join(self.scenes_dir, self.scenes_list[scene], "4.tiff"),
                             os.path.join(self.scenes_dir, self.scenes_list[scene], "7.tiff"))
            
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]


    def __getitem__(self, index):
        # Read exposure times
        # expoTimes = read_expo_times(self.image_list[index][0])
        expoTimes = [1, 4, 16]
        # Read LDR images
        ldr_images = read_images_custom(self.image_list[index][1])
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)
        
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        

        sample = {"input0": img0, "input1": img1, "input2": img2, "label": img1[3:6,:,:]}
        return sample

    def __len__(self):
        return len(self.scenes_list)
