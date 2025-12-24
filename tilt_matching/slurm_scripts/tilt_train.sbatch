#!/bin/bash
#SBATCH --job-name=tilt_sudoku
#SBATCH --account=albergo_lab
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yuyuanchen@math.harvard.edu

OUTPUT_DIR="/n/home06/yuyuan0/TM_SPG/tilt_matching/outputs/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate /n/home06/yuyuan0/conda/envs/spg

# -------------------------
# Networking configuration
# -------------------------
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
# Debug prints (per node)
# -------------------------
echo "MASTER_NODE=$MASTER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "==== IPv4 addrs on $(hostname) ===="
ip -o -4 addr show | awk '{print $2, $4}'
echo "==================================="

# Print env from every node once (helps debugging)
srun --ntasks-per-node=1 bash -lc \
  'echo "HOST=$(hostname) NODEID=$SLURM_NODEID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT GLOO_IF=$GLOO_SOCKET_IFNAME NCCL_IF=$NCCL_SOCKET_IFNAME"'

# -------------------------
# Launch distributed job
# -------------------------
LOGDIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGDIR"

cd /n/home06/yuyuan0/TM_SPG/tilt_matching


srun --ntasks-per-node=1 --gpus-per-task=4 \
  python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    --log_dir="$LOGDIR" \
    --redirects 3 \
    --tee 3 \
    tilt_train.py \
    --config-path /n/home06/yuyuan0/TM_SPG/tilt_matching/config \
    --config-name tilt_matching.yaml
