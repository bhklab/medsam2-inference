#!/bin/bash
#SBATCH --job-name=medsam-inference
#SBATCH --mem=84G
#SBATCH --gres=gpu:v100:1
#SBATCH -t 2-23:59:59
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH -C "gpu32g"
#SBATCH --mail-user=katy.scott@uhn.ca
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output="/cluster/projects/radiomics/Temp/katy/slurm_logs/medsam2-inference/%A-%x.out"


DATASET="RADCURE_GTVp"
SUBSET="" # put underscore before subset for file names to look nice. Leave blank if not subsetting.
DATA_CSV_NAME="mit_${DATASET}${SUBSET}_medsam_input_apptainer.csv"

apptainer exec --nv \
--mount type=bind,source=/cluster/projects/radiomics/,destination=/hostopt/ \
medsam-inference.sif \
python inference.py \
/hostopt/Projects/medsam2-inference/data/procdata/$DATASET/metadata/$DATA_CSV_NAME \
/hostopt/Projects/medsam2-inference/data/results/$DATASET$SUBSET \
--window_level 20 \
--window_width 400 \
--overlay_bbox True \
