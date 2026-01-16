#!/bin/bash
#SBATCH --job-name=eval_gsm8k
#SBATCH --account=albergo_lab
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=[h100|h200]
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB
#SBATCH --time=0:17:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

cd /n/home06/yuyuan0/TM_SPG/eval
OUTPUT_DIR="/n/home06/yuyuan0/TM_SPG/eval/eval_tilt/output_random_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO

srun --ntasks-per-node=1 --gpus-per-task=1 \
  torchrun \
    --standalone \
    --nproc_per_node=1 \
    eval.py \
    --dataset "gsm8k" \
    --batch_size 8 \
    --gen_length 256 \
    --output_dir "$OUTPUT_DIR" \
    --model_path "/n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct" \
    --temperature 0.0 \
    --seed 42 \
    --diffusion_steps 128 \
    --remasking "low_confidence" \
    --num_prompts_gsm 256 \
    --adapter "student" \
    --checkpoint_path "/n/netscratch/albergo_lab/Everyone/frank/llada_tm/gsm8k_long_noconf/checkpoint-a-2.500-2.0.ckpt"