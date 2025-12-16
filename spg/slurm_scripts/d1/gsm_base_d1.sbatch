#!/bin/bash
#SBATCH --output=../../slurm/%A/slurm.out
#SBATCH --job-name=gsm-d1
#SBATCH --time=72:00:00
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=50
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

export LOGDIR=./logs
mkdir -p $LOGDIR
echo $LOGDIR


DATASET="gsm8k"
RUN_NAME=${DATASET}_base_d1
NUM_ITER=4
srun --output ${LOGDIR}/d1_%j.out \
    accelerate launch \
        --config_file ../accelerate_genai_a100.yaml \
        --main_process_port 12346 ../../diffu_grpo_train.py \
        --config ../train.yaml \
        --model_path /n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --trainer diffu_grpo \
        --num_generations 6 \
        --per_device_train_batch_size 6 \
        --gradient_accumulation_steps 2 \
        --output_dir ./output_dir/$RUN_NAME