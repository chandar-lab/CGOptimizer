#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=4                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                    # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --array=0-7
#SBATCH --output=/home/pamcrae/job_output.txt
#SBATCH --error=/home/pamcrae/job_error.txt

# 1. Create your environement locally
module load python/3.6
source $HOME/.env/crit-grad/bin/activate

#cp -r $HOME/CriticalGradientOptimization/Dataset $SCRATCH/

python cifar-wandb.py --data_path ./Dataset --results_path .

#cp -r $SLURM_TMPDIR/Results/mnist $HOME/CriticalGradientOptimization/Results
