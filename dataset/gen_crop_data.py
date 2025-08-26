import os
import glob
import shutil
import argparse
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def get_croped_data_per_scene(scene_dir, patch_size=128, stride=64):
    exposure_file_path = os.path.join(scene_dir, "exposure.txt")
    ldr_file_path = sorted(glob.glob(os.path.join(scene_dir, "*.tif")))
    label_path = os.path.join(scene_dir, "HDRImg.hdr")

    ldr_0 = cv2.imread(ldr_file_path[0], cv2.IMREAD_UNCHANGED)
    ldr_1 = cv2.imread(ldr_file_path[1], cv2.IMREAD_UNCHANGED)
    ldr_2 = cv2.imread(ldr_file_path[2], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    
    crop_data = []
    h, w, _ = label.shape
    for x in range(w):
        for y in range(h):
            if x * stride + patch_size <= w and y * stride + patch_size <= h:
                crop_sample = {
                    "ldr_0": ldr_0[
                        y * stride : y * stride + patch_size,
                        x * stride : x * stride + patch_size,
                    ],
                    "ldr_1": ldr_1[
                        y * stride : y * stride + patch_size,
                        x * stride : x * stride + patch_size,
                    ],
                    "ldr_2": ldr_2[
                        y * stride : y * stride + patch_size,
                        x * stride : x * stride + patch_size,
                    ],
                    "label": label[
                        y * stride : y * stride + patch_size,
                        x * stride : x * stride + patch_size,
                    ],
                    "exposure_file_path": exposure_file_path,
                }
                
                if np.mean(crop_sample["ldr_1"]) > 60000:
                    continue

                crop_data.append(crop_sample)
    print(f"{len(crop_data)} samples of scene {scene_dir}")
    return crop_data


def rotate_sample(data_sample, mode=0):
    flag = cv2.ROTATE_90_CLOCKWISE if mode == 0 else cv2.ROTATE_90_COUNTERCLOCKWISE
    return {
        k: cv2.rotate(v, flag) if "ldr" in k or "label" in k else v
        for k, v in data_sample.items()
    }


def flip_sample(data_sample, mode=0):
    return {
        k: cv2.flip(v, mode) if "ldr" in k or "label" in k else v
        for k, v in data_sample.items()
    }


def save_sample(data_sample, save_root, id):
    save_path = os.path.join(save_root, id)
    os.makedirs(save_path, exist_ok=True)
    shutil.copyfile(
        data_sample["exposure_file_path"], os.path.join(save_path, "exposure.txt")
    )
    cv2.imwrite(os.path.join(save_path, "0.tif"), data_sample["ldr_0"])
    cv2.imwrite(os.path.join(save_path, "1.tif"), data_sample["ldr_1"])
    cv2.imwrite(os.path.join(save_path, "2.tif"), data_sample["ldr_2"])
    cv2.imwrite(os.path.join(save_path, "label.hdr"), data_sample["label"])


def process_scene(scene_dir, cropped_training_data_path, patch_size, stride, aug):
    croped_data = get_croped_data_per_scene(scene_dir, patch_size, stride)
    local_counter = 0
    for data in croped_data:
        base_id = f"{os.path.basename(scene_dir)}_{str(local_counter).zfill(5)}"
        save_sample(data, cropped_training_data_path, base_id)
        local_counter += 1

        if aug:
            save_sample(
                rotate_sample(data, 0), cropped_training_data_path, base_id + "_r90"
            )
            save_sample(
                flip_sample(data, 0), cropped_training_data_path, base_id + "_fv"
            )
    return f"{scene_dir} done."


def main():
    parser = argparse.ArgumentParser(description="Prepare cropped data")
    parser.add_argument(
        "--data_root", type=str, default="/home/urso/Datasets/Tel"
    )
    parser.add_argument("--patch_size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--aug", action="store_true", default=True)
    args = parser.parse_args()

    full_size_training_data_path = os.path.join(args.data_root, "Training")
    cropped_training_data_path = os.path.join(
        args.data_root, f"sig17_training_crop{args.patch_size}_stride{args.stride}_custom"
    )
    os.makedirs(cropped_training_data_path, exist_ok=True)

    scene_dirs = sorted(glob.glob(os.path.join(full_size_training_data_path, "*")))

    print(f"Procesando {len(scene_dirs)} escenas usando {cpu_count()} núcleos...")

    with Pool(cpu_count()) as pool:
        results = pool.map(
            partial(
                process_scene,
                cropped_training_data_path=cropped_training_data_path,
                patch_size=args.patch_size,
                stride=args.stride,
                aug=args.aug,
            ),
            scene_dirs,
        )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
