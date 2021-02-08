#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=4                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                    # Ask for 1 GPU
#SBATCH --mem=16G                        # Ask for 32 GB of RAM
#SBATCH --array=0-7
#SBATCH --output=/home/pamcrae/out.txt
#SBATCH --error=/home/pamcrae/err.txt

# 1. Create your environement locally
module load python/3.6
source $HOME/.env/crit-grad/bin/activate

cp -r /home/pamcrae/CriticalGradientOptimization/Dataset $SCRATCH/

python ptb-wandb.py --data_path $SCRATCH/Dataset --results_path .

#cp -R $SLURM_TMPDIR/wandb /project/6004852/pamcrae/CriticalGradientOptimization
