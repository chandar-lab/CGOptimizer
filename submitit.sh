#!/bin/bash
# Parameters
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/pamcrae/CriticalGradientOptimization/dumps/projects/crit-grad/nli-task/2021-01-22_rand_eval_snli_InferSent/submitit_logs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=submitit
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/pamcrae/CriticalGradientOptimization/dumps/projects/crit-grad/nli-task/2021-01-22_rand_eval_snli_InferSent/submitit_logs/%j_0_log.out
#SBATCH --signal=USR1@90
#SBATCH --time=10
#SBATCH --wckey=submitit
# command
export SUBMITIT_EXECUTOR=slurm
srun --output '/scratch/pamcrae/CriticalGradientOptimization/dumps/projects/crit-grad/nli-task/2021-01-22_rand_eval_snli_InferSent/submitit_logs/%j_%t_log.out' --error '/scratch/pamcrae/CriticalGradientOptimization/dumps/projects/crit-grad/nli-task/2021-01-22_rand_eval_snli_InferSent/submitit_logs/%j_%t_log.err' --unbuffered /home/pamcrae/.envs/crit-grad/bin/python -u -m submitit.core._submit '/scratch/pamcrae/CriticalGradientOptimization/dumps/projects/crit-grad/