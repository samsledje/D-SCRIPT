#!/bin/bash
#SBATCH --job-name=test_bernett
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_test_%a.out
#SBATCH --error=logs/%A_test_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
module purge
module load ngc/1.0
module load anaconda/2024.10
#module load pytorch/2.5.1-cuda12.1-cudnn9


module purge   
hash -r

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dscript



export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r



TOPSY_TURVY=
EMBEDDING_DIR=/cluster/tufts/cowenlab/tt3d+/data/esm2/bernett
SEQ_DIR=/cluster/tufts/cowenlab/tt3d+/data/gold_data
FOLDSEEK_FASTA=/cluster/tufts/cowenlab/tt3d+/data/foldseek_files/bernett.fasta
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
MODEL_PARAMS=""
DEVICE=0
# Set default model here:
#MODEL=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/results/bernett_esm2_train_diex5e3/bernett_diex5e3_epoch04.sav
MODEL=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/results/bernett_esm2_no/bernett_ep__epoch06.sav
usage() {
    echo "USAGE: ./test.sh [-d DEVICE] [-m MODEL] [-T MODEL_TYPE]

    OPTIONS:

    -d DEVICE: Device used
    -m MODEL: The model sav file
    -T: Set this flag if if the model passed by the '-m MODEL' command is a TT3D Model. Unset this for Topsy-Turvy/D-SCRIPT
    "
}


while getopts "d:m:T" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        m) MODEL=${OPTARG}
        ;;
        T) MODEL_PARAMS="${MODEL_PARAMS} --add_foldseek_after_projection --foldseek_vocab ${FOLDSEEK_VOCAB} --foldseek_fasta ${FOLDSEEK_FASTA} --allow_foldseek"
        ;;
        *) usage
        exit 1;
    esac
done

# Construct the folder
OUTPUT_FLD=${MODEL%/*}
OUTPUT_FILE=${MODEL##*/}
OUTPUT_FILE_PREF=${OUTPUT_FILE%.*}
OUTPUT_FOLDER=${OUTPUT_FLD}/cmap-${OUTPUT_FILE_PREF}

echo "Output folder: ${OUTPUT_FOLDER}, model: ${MODEL}, DEVICE: ${DEVICE}"
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir $OUTPUT_FOLDER; fi


EMBEDDING=${EMBEDDING_DIR}
TEST=${SEQ_DIR}/bernett_test.tsv
OP_FOLDER_ORG=${OUTPUT_FOLDER}/
if [ ! -d ${OP_FOLDER_ORG} ]; then mkdir -p ${OP_FOLDER_ORG}; fi
OP_FILE=${OP_FOLDER_ORG}/${OUTPUT_FILE}
python -u -m dscript.commands.predict_serial_ori --pairs ${TEST} --model ${MODEL} --embeddings ${EMBEDDING}  -d $DEVICE -o $OP_FILE --store_cmaps

