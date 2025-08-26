#!/bin/bash
#SBATCH --job-name=TestPavicHDR 
#SBATCH --output=/home/urso/FLASH/logs/latest.out 
#SBATCH --error=/home/urso/FLASH/logs/latest.out  
#SBATCH --partition=gpu      
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G            
#SBATCH --cpus-per-task=16   



export SLURM_OVERLAP=yes
source '/opt/intel/oneapi/setvars.sh' 
export CONDA_ENV=pytorch-gpu 

module try-load singularity
echo job start time is `date`

exec_node=${SLURM_JOB_NODELIST}
cpu_curr_node=${SLURM_CPUS_ON_NODE}
if [ "${cpu_curr_node}" = "" ]; then  cpu_curr_node={{ cores_per_node|default:1 }}; fi

set -x
srun -N1 -n1 --cpu_bind=cores -l --nodelist=${exec_node} \
    --cpus-per-task=${cpu_curr_node} \
    singularity exec --nv \
    -B /home/urso -B /scratch \
    --pwd /home/urso//home/urso/FLASH \
    /home/urso/.lico/container/Python.sif bash -c "
    
    . /home/urso/PavicHDR/venv/bin/activate

    python /home/urso/FLASH/test.py \
      --dataset_dir /home/urso/FLASH/dataset/Tursun \
      --loader Tursun \
      --ckpt /home/urso/FLASH/runs/train_32/best_checkpoint.pth \
      # --isTLC

    
    " 
set +x


echo job end time is `date`