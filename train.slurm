#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100
#SBATCH --job-name=rl-training
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem=256G

date
nvidia-smi
module purge
module load singularity

singularity exec --bind `pwd`:/home --pwd /home --nv images/gym.sif python mario.py