#!/bin/bash
#SBATCH --time 48:00:00
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output /home/sakkaya/std_unet_gs/best_with_smaller_filters/wire/out_filter_size_rmsprop_bfl_wire.txt
#SBATCH --error /home/sakkaya/std_unet_gs/best_with_smaller_filters/wire/err_filter_size_rmsprop_bfl_wire.txt
#SBATCH --job-name=python-rail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s289332@studenti.polito.it
#SBATCH --partition=cuda

module load nvidia/cudasdk/10.1

source ~/.bashrc

conda activate rail

python /home/sakkaya/std_unet_gs/best_with_smaller_filters/wire/filter_size_rmsprop_bfl_wire.py

