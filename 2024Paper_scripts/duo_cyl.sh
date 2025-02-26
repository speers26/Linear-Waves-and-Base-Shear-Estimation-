#!/bin/bash

#SBATCH -J duo_cyl-job
#SBATCH -c 76
#SBATCH -o duo_cyl.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.speers@lancaster.ac.uk
#SBATCH --mem 200000

source ~/start-pyenv
source /beegfs/client/default/speersm/force_calculation_and_wave_sim/.venv/bin/activate
export OMP_NUM_THREADS=1
srun python dist_write_parallel.py
