dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="${dir}/data/AMR"
dataset_dir="${data_dir}/amr_annotation_3.0"
dev_dir="${dataset_dir}/data/amrs/split/dev"
test_dir="${dataset_dir}/data/amrs/split/test"
train_dir="${dataset_dir}/data/amrs/split/training"

python3 "${dir}/scripts/preprocess_amr.py" --input-folder "${dev_dir}" --output-folder "${data_dir}/dev/en"
python3 "${dir}/scripts/preprocess_amr.py" --input-folder "${test_dir}" --output-folder "${data_dir}/test/en"
python3 "${dir}/scripts/preprocess_amr.py" --input-folder "${train_dir}" --output-folder "${data_dir}/training/en"

rm "${dataset_dir}"