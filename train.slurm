#!/bin/bash

#SBATCH --job-name=PavicHDR
#SBATCH --output=/home/urso/PavicHDR/logs/latest_train1.out 
#SBATCH --error=/home/urso/PavicHDR/logs/latest_train1.out  

#SBATCH --partition=gpu      
#SBATCH --gres=gpu:1
#SBATCH --mem=16G            
#SBATCH --cpus-per-task=8


singularity exec --nv /home/urso/.lico/container/Python.sif /bin/bash -c "

  source /home/urso/PavicHDR/venv/bin/activate

  python /home/urso/PavicHDR/train.py \
    --dataset_dir /home/urso/Datasets/Kalantari/ \
    --sub_set sig17_training_crop500_stride250 \
    --batch_size 2 \
    --logdir /home/urso/PavicHDR/runs_4/train_37 \
    --epochs 10000 \
    --val_epochs 1 \
    --lr0 2e-4\
    --lrf 1e-4 \
    --lr_decay_interval 100 \
    # --resume /home/urso/PavicHDR/runs_4/train_35/best_checkpoint.pth \
    "