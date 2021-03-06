#!/bin/bash

GPU_ID=${1:-0}
NET=${2:-"ZF"}
STAGES=${3:-3}
DATA_DIR=${4:-"/srv/caffe-data/datasets/boxes_family_gray"}
TASK=${5:-"seg"}
MODEL=${6:-"output/boxes_family_gray/zf_mnc_3stage_iter_20.caffemodel.h5"}

# Remove slashes at the end of the path.
DATA_DIR=${DATA_DIR%/}

# Get dataset name.
DATASET=${DATA_DIR##*/}

# Prompt cache removal.
read -p "Remove cache? [y/N] " yn
case $yn in
  [Yy]* ) rm -rf cache/*; [[ -d "output/${DATASET}" ]] && find output/${DATASET} -name "*.pkl" -type f -delete;
esac

# Compute the number of classes based on the classes.txt file.
if [ -f ${DATA_DIR}/classes.txt ]; then
	NUM_CLASSES=$(($(cat ${DATA_DIR}/classes.txt | wc -l) + 1))
else
	echo "Could not find file 'classes.txt' in the data directory, aborting."
	exit -1
fi

# Generate prototxt file.
python experiments/scripts/generate_prototxt.py \
	${NET}/mnc_${STAGES}stage/test.prototxt.template output/${DATASET}/test.prototxt -p num_classes=${NUM_CLASSES}

# Start testing.
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def output/${DATASET}/test.prototxt \
  --net ${MODEL} \
  --imdb "path" \
  --data-dir ${DATA_DIR} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml \
  --task ${TASK}
