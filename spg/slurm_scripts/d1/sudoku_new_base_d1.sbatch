#!/bin/bash
#SBATCH --output=/n/home06/yuyuan0/TM_SPG/output_logs/%A/slurm.out
#SBATCH --job-name=gsm-d1
#SBATCH --partition=gpu
#SBATCH --account=albergo_lab
#SBATCH --time=3:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

### Small debug prints (will go into slurm log)
echo "Active conda envs:"
conda info --envs
echo "Python:     $(which python)"
echo "Accelerate: $(which accelerate)"

export LOGDIR=./logs
mkdir -p $LOGDIR
echo $LOGDIR

NUM_ITER=4
FEW_SHOT=3
DATASET="sudoku_new"
RUN_NAME=${DATASET}_${FEW_SHOT}shot_base_d1

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
        --per_device_train_batch_size 3 \
        --gradient_accumulation_steps 8 \
        --output_dir ./output_dir/$RUN_NAME \
        --temperature 0.3 \
        --few_shot $FEW_SHOT \
        --max_prompt_length 1500