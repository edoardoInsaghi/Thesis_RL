#!/bin/bash

#SBATCH --job-name=turtles_RL
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --time=2:00:00
#SBATCH --partition=GPU
#SBATCH --gpus=1

source ~/Thesis_RL/venv/bin/activate
python3 shared_main2.py
