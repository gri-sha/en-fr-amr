SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/AMR"

DATASET_DIR="${SCRIPT_DIR}/data/amr_annotation_3.0"
DEV_DIR="${SCRIPT_DIR}/data/amr_annotation_3.0/data/amrs/split/dev"
TEST_DIR="${SCRIPT_DIR}/data/amr_annotation_3.0/data/amrs/split/test"
TRAIN_DIR="${SCRIPT_DIR}/data/amr_annotation_3.0/data/amrs/split/training"

python3 "${SCRIPT_DIR}/scripts/preprocess_amr.py" --input-folder "${DEV_DIR}" --output-folder "${DATA_DIR}/dev/en"
python3 "${SCRIPT_DIR}/scripts/preprocess_amr.py" --input-folder "${TEST_DIR}" --output-folder "${DATA_DIR}/test/en"
python3 "${SCRIPT_DIR}/scripts/preprocess_amr.py" --input-folder "${TRAIN_DIR}" --output-folder "${DATA_DIR}/training/en"