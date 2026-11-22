#!/usr/bin/env bash
# train.sh NAME GPUS [sim-train options...]
# Data-parallel PPO on GPUS GPUs of this machine (torchrun). Trains on
# data/nuplan/train_full and evaluates Val14, Test14-random and Test14-hard in our
# simulator (NR and R) during training. Output: outputs/train/NAME/{train.log,
# metrics.jsonl, config.json, checkpoints/}. The log ends with "EXIT <code>".
#
# The report's runs (8 GPUs each; --worlds is per GPU):
#   scripts/train.sh replay      8 --worlds 2048 --rollouts-per-update 4 --epochs 47 --eval-updates 15
#   scripts/train.sh selfplay    8 --worlds 512 --rollouts-per-update 2 --epochs 9 --eval-updates 20 \
#       --log-worlds 0 --self-play-worlds 1
#   scripts/train.sh selfplay_dr 8 --worlds 512 --rollouts-per-update 2 --epochs 9 --eval-updates 20 \
#       --log-worlds 0 --self-play-worlds 1 --domain-randomization --model reward_conditions=true
set -u
name=$1; gpus=$2; shift 2
out=outputs/train/$name
mkdir -p "$out"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run torchrun --standalone --nproc_per_node "$gpus" -m cs5478_project.training.train \
    --train data/nuplan/train_full \
    --eval val14=data/nuplan/val14 --eval test14=data/nuplan/test14 \
    --eval test14_hard=data/nuplan/test14_hard --eval-reactive \
    --output "$out" --no-cache-scenes --store-on-host --cpu-threads 4 \
    "$@" > "$out/train.log" 2>&1
echo "EXIT $?" >> "$out/train.log"
