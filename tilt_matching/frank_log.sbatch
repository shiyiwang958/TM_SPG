#!/bin/bash
#SBATCH --job-name=tilt_countdown
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=fwang@math.harvard.edu
#SBATCH --output=/n/home03/sw958/project_1/SPG/tilt_matching/outputs/output_%j.out

OUTPUT_DIR="/n/home03/sw958/project_1/SPG/tilt_matching/outputs/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
mamba deactivate
mamba activate spg
module load cuda/12.4.1-fasrc01

# -------------------------
# Networking configuration
# -------------------------
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_USE_IPV6=0

# Pick default IPv4 route interface for rendezvous (ethernet)
DEFAULT_IF=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)
if [ -z "${DEFAULT_IF:-}" ]; then
  echo "ERROR: could not determine default IPv4 interface"
  ip -o -4 route show
  exit 2
fi
export GLOO_SOCKET_IFNAME="$DEFAULT_IF"

# NCCL data plane over InfiniBand (ib0)
export NCCL_SOCKET_IFNAME=ib0

# Choose a fixed port first (avoids "random port blocked" issues)
export MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 50000 ))

export NODE_RANK=$SLURM_NODEID
export NNODES=$SLURM_JOB_NUM_NODES

# Rank0 host
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}" | head -n 1)

# Share MASTER_ADDR to all nodes via a shared file
MASTER_FILE="$OUTPUT_DIR/master_addr.txt"

if [ "$SLURM_NODEID" -eq 0 ]; then
  # Pick em2 IPv4 on rank0 as MASTER_ADDR (rendezvous address)
  MASTER_ADDR=$(ip -o -4 addr show dev "$DEFAULT_IF" | awk '{print $4}' | cut -d/ -f1 | head -n 1 || true)
  if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: could not find IPv4 on $DEFAULT_IF for rank0"
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

cd /n/home03/sw958/project_1/SPG/tilt_matching


srun --ntasks-per-node=1 --gpus-per-task=2 \
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    --log_dir="$LOGDIR" \
    --redirects 3 \
    --tee 3 \
    train_log.py \
    --config-path /n/home03/sw958/project_1/SPG/tilt_matching/config \
    --config-name log_countdown.yaml
