#!/bin/bash
#SBATCH --output=/n/home06/yuyuan0/TM_SPG/output_logs/%A/slurm.out
#SBATCH --job-name=tm-sudoku
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --time=6:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

export LOGDIR=./logs
mkdir -p $LOGDIR
echo $LOGDIR

srun --ntasks-per-node=1 --gpus-per-task=4 \
  torchrun \
    --standalone \
    --nproc_per_node=4 \
    /n/home06/yuyuan0/TM_SPG/tilt_matching/tilt_train.py \
    --config-path /n/home06/yuyuan0/TM_SPG/tilt_matching/config \
    --config-name tilt_matching