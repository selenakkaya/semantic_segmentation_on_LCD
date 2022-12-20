#!/bin/bash
#SBATCH --time 48:00:00
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output /home/sakkaya/UNetPlus/out.txt
#SBATCH --error /home/sakkaya/UNetPlus/err.txt
#SBATCH --job-name=python-rail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s289332@studenti.polito.it
#SBATCH --partition=cuda
module load nvidia/cudasdk/10.1
module load intel/python/3
source RAIL/bin/activate
python /home/sakkaya/binary-segmentation/unet_wire.py 