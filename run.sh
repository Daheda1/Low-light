#!/bin/bash
#SBATCH --job-name=singularity_test
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=60G
#SBATCH --gres=gpu:10

singularity exec --nv /ceph/container/pytorch_24.03-py3.sif python Train.py
