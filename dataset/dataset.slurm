#!/bin/bash

#SBATCH --job-name=datasethdr 
#SBATCH --output=/home/urso/PavicHDR/logs/latest_dataset.out 
#SBATCH --error=/home/urso/PavicHDR/logs/latest_dataset.out  
#SBATCH --partition=gpu      
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G            
#SBATCH --cpus-per-task=60
#SBATCH --nodelist=gn03      

singularity exec --nv /home/urso/.lico/container/Python.sif /bin/bash -c "

 . /home/urso/PavicHDR/venv/bin/activate

 python /home/urso/PavicHDR/dataset/gen_crop_data.py \
    --data_root /home/urso/Datasets/Kalantari \
    --patch_size 128 \
    --stride 64 \
    --aug
"