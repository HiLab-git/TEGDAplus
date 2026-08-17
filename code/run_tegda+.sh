#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method tegda_plus --exp BraTs2023_GLI2PED_TEGDAplus
python test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method tegda_plus --exp MMS_A2B_TEGDAplus
python test_time_adaptation_2D_online_eval.py --target_domain C --TTA_method tegda_plus --exp MMS_A2C_TEGDAplus
python test_time_adaptation_2D_online_eval.py --target_domain D --TTA_method tegda_plus --exp MMS_A2D_TEGDAplus
