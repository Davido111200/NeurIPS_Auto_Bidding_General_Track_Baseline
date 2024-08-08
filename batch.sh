#!/bin/bash
#SBATCH --job-name=baseline_comp
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --qos=batch
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source ~/anaconda3/bin/activate /home/Projects/a2i2rl/NeurIPS_Auto_Bidding_General_Track_Baseline/nips-bidding-env

python main/main_iql.py &
python main/main_bc.py & 
python main/main_onlineLp.py &

wait