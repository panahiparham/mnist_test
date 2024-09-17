#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-1
#SBATCH --output=outputs/MNIST_on_cpu.txt


source ~/virtual_env_meta_step/bin/activate
module load python/3.11
#module load cuda

python3 train_mnist.py 

