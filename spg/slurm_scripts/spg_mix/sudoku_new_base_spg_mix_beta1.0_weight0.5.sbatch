#!/bin/bash
#SBATCH --job-name=sudoku_spg_mix
#SBATCH --account=albergo_lab
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=3:30:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

OUTPUT_DIR="/n/home06/yuyuan0/TM_SPG/spg/outputs/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

# -------------------------
# Networking configuration
# -------------------------
export SPG_DEBUG_ATTN=1
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_USE_IPV6=0

# Rendezvous (TCPStore/GLOO) over Ethernet (em2)
export GLOO_SOCKET_IFNAME=em2

# NCCL data plane over InfiniBand (ib0)
export NCCL_SOCKET_IFNAME=ib0

# Choose a fixed port first (avoids "random port blocked" issues)
export MASTER_PORT=29500

export NODE_RANK=$SLURM_NODEID
export NNODES=$SLURM_JOB_NUM_NODES

# Rank0 host
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}" | head -n 1)

# Share MASTER_ADDR to all nodes via a shared file
MASTER_FILE="$OUTPUT_DIR/master_addr.txt"

if [ "$SLURM_NODEID" -eq 0 ]; then
  # Pick em2 IPv4 on rank0 as MASTER_ADDR (rendezvous address)
  MASTER_ADDR=$(ip -o -4 addr show dev em2 | awk '{print $4}' | cut -d/ -f1 || true)
  if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: could not find IPv4 on em2 for rank0"
    ip -o -4 addr show
    exit 2
  fi
  echo "$MASTER_ADDR" > "$MASTER_FILE"
fi

# Wait for rank0 to write the addr
while [ ! -s "$MASTER_FILE" ]; do sleep 1; done
export MASTER_ADDR=$(cat "$MASTER_FILE")

# -------------------------
# Launch distributed job
# -------------------------
LOGDIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGDIR"

# Generate a random port number between 10000 and 65535
RANDOM_PORT=$((RANDOM % 55536 + 10000))
DATASET="sudoku_new"
FEW_SHOT=3
RUN_NAME=${DATASET}_${FEW_SHOT}shot_base_spg_mix_beta1.0_weight0.5
NUM_ITER=4
srun --cpu-bind=none --output ${LOGDIR}/spg_mix_%j.out \
    accelerate launch \
        --config_file ../accelerate_genai_a100.yaml \
        --main_process_port $RANDOM_PORT ../../diffu_grpo_train.py \
        --config ../train.yaml \
        --model_path /n/netscratch/albergo_lab/Everyone/frank/hf_models/LLaDA-8B-Instruct \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir /n/home06/yuyuan0/TM_SPG/spg/$RUN_NAME \
        --trainer spg \
        --forward_type block_random \
        --num_t 2 \
        --min_t 0 \
        --max_t 1 \
        --num_generations 6 \
        --per_device_train_batch_size 3 \
        --gradient_accumulation_steps 4 \
        --beta 0.0 \
        --logp_estimation mix \
        --mix_weight 0.5 \
        --eubo_beta 1.0 \
        --temperature 0.3 \
        --few_shot $FEW_SHOT \
        --max_prompt_length 1500