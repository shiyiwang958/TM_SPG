#!/bin/bash

# where to save
TARGET_DIR="/n/home03/sw958/project_1/SPG/hf_models"

python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="GSAI-ML/LLaDA-8B-Instruct",
    local_dir="${TARGET_DIR}"
)
EOF