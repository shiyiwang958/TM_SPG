#!/bin/bash
#SBATCH --output=/n/home06/yuyuan0/TM_SPG/output_logs/%A/slurm.out
#SBATCH --job-name=gsm-d1
#SBATCH --partition=gpu
#SBATCH --account=albergo_lab
#SBATCH --time=1:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

export LOGDIR=./logs
mkdir -p $LOGDIR
echo $LOGDIR

source activate spg

SAVE_DIR=/fsx-checkpoints/

DATASET="gsm8k"
RUN_NAME=${DATASET}_base_d1
MODEL_PATH=${SAVE_DIR}/hf_models/LLaDA-8B-Instruct
NUM_ITER=4
srun --output ${LOGDIR}/d1_%j.out \
    accelerate launch \
        --config_file ../accelerate_genai_a100.yaml \
        --main_process_port 12346 ../../diffu_grpo_train.py \
        --config ../train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --trainer diffu_grpo \
        --num_generations 6 \
        --per_device_train_batch_size 6 \
        --gradient_accumulation_steps 2 \
        --output_dir ${SAVE_DIR}/spg/$RUN_NAME