#!/bin/bash
#SBATCH --job-name=eval_sudoku
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --output=../logs_eval/eval_sudoku_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=fwang@math.harvard.edu


source activate spg

# TODO: Change to eval only one checkpoint.
# SUBMIT THIS IN THE EVAL DIRECTORY!
# Print environment info for debugging
echo "Python version: $(python --version)"
echo "PyTorch installation check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"
which torchrun || echo "torchrun not found in PATH"

# Configuration variables
# GPU_IDS will be automatically set by SLURM, but we'll use all available GPUs
GPU_IDS=(0 1 2 3)

# Generate a random port number between 10000 and 65535
MASTER_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $MASTER_PORT"

# Arrays of tasks and generation lengths
TASKS=("sudoku")
GEN_LENGTHS=(128 256 512)
SAVE_DIR=/fsx-checkpoints

# no checkpoints to loop over

# Use SLURM allocated GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  # If SLURM has set CUDA_VISIBLE_DEVICES, use those GPUs
  IFS=',' read -ra SLURM_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  GPU_IDS=("${SLURM_GPUS[@]}")
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do  
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
      
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --few_shot 3 \
      --output_dir "tilt_results/soduku_tilt" \
      --model_path "/n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct" \
      --checkpoint_path "${SAVE_DIR}/spg/sudoku_new_3shot_base_spg_mix_beta1.0/checkpoint"
      # TODO: change to your checkpoint path
  done
done


echo "All evaluations completed!" 