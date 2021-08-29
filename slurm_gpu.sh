#!/bin/bash
#SBATCH --job-name=kb_sbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32GB
#SBATCH --output=job_%j.out
#SBATCH -e joberror_%j.err
#SBATCH --time=01:00:00          # HH:MM:SS
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=faton.rekathati@kb.se

srun hostname
python train.py