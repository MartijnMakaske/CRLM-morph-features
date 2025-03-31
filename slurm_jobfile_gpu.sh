#!/bin/sh

# for more details on slurm scripts, check https://amsterdamumc.sharepoint.com/sites/mip_wiki/SitePages/slurm.aspx

#SBATCH --job-name=test-gpu                     # a convenient name for your job
#SBATCH --gres=gpu:1g.10gb:1                    # number of GPU you want to use
#SBATCH --mem=8G                                # max memory used
#SBATCH --partition=luna-short                  # using luna-long queue for long period > 8h or luna-short for short < 8h
#SBATCH --cpus-per-task=1                       # max CPU cores per process
#SBATCH --time=0-04:00                          # time limit (DD-HH:MM)
#SBATCH --nice=10000                             # allow other priority jobs to go first (note, this is different from the linux nice command below)

set -eu                                         # Exit immediately on error or on undefined variable

                                # Replace with command line start of your script (example: python3 /home/rnga/khrus/myscript.py)

/scratch/bmep/mfmakaske/session/model_training.py