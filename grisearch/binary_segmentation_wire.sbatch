#!/bin/bash
#SBATCH --time 48:00:00
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output /home/sakkaya/grid_search/out_unet_wire.txt
#SBATCH --error /home/sakkaya/grid_search/err_unet_wire.txt
#SBATCH --job-name=python-rail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s289332@studenti.polito.it
#SBATCH --partition=cuda
module load nvidia/cudasdk/10.1

source ~/.bashrc

conda activate rail



python /home/sakkaya/grid_search/grid_search_binary_wire.py