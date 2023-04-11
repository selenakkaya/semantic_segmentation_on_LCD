#!/bin/bash
#SBATCH --time 48:00:00
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output /home/sakkaya/std_unet_gs/best_with_smaller_filters/mesh/out_filter_size_adam_lr_0001_bce_mesh.txt
#SBATCH --error /home/sakkaya/std_unet_gs/best_with_smaller_filters/mesh/err_filter_size_adam_lr_0001_bce_mesh.txt
#SBATCH --job-name=python-rail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s289332@studenti.polito.it
#SBATCH --partition=cuda

module load nvidia/cudasdk/10.1

source ~/.bashrc

conda activate rail

python /home/sakkaya/std_unet_gs/best_with_smaller_filters/mesh/filter_size_adam_bce_mesh.py

        