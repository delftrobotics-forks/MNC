#!/bin/bash

GPU_ID=${1:-0}
NET=${2:-"ResNet50"}
STAGES=${3:-3}
DATA_DIR=${4:-/srv/caffe-data/datasets/meyn-complete}
ITERS=${5:-1000}

# Remove slashes at the end of the path.
DATA_DIR=${DATA_DIR%/}

# Get dataset name.
DATASET=${DATA_DIR##*/}

# Prompt cache removal.
read -p "Remove cache? [y/N] " yn
case $yn in
	[Yy]* ) rm -rf cache/*; rm -rf data/cache/*;
esac

# Prompt old models removal.
read -p "Remove old models? [y/N] " yn
case $yn in
	[Yy]* ) rm -rf output/${DATASET}*;
esac


# Compute the number of classes based on the classes.txt file.
if [ -f ${DATA_DIR}/classes.txt ]; then
	NUM_CLASSES=$(($(cat ${DATA_DIR}/classes.txt | wc -l) + 1))
else
	echo "Could not find file 'classes.txt' in the data directory, aborting."
	exit -1
fi

# Prepare folder for storing the prototxt files.
[ ! -d output/${DATASET} ] && mkdir -p output/${DATASET}

# Generate prototxt files.
python experiments/scripts/generate_prototxt.py \
	${NET}/mnc_${STAGES}stage/train.prototxt.template  output/${DATASET}/train.prototxt  -p num_classes=${NUM_CLASSES}

python experiments/scripts/generate_prototxt.py \
	${NET}/mnc_${STAGES}stage/solver.prototxt.template output/${DATASET}/solver.prototxt -p dataset_name=${DATASET}

# Get path for Caffe model.
case $NET in
	ZF)
		NET_INIT=data/imagenet_models/${NET}.v2.caffemodel
		;;
	ResNet50)
		NET_INIT=data/imagenet_models/ResNet-50-model.caffemodel
		;;
	ResNet101)
		NET_INIT=data/imagenet_models/ResNet-101-model.caffemodel
		;;
	*)
		NET_INIT=data/imagenet_models/${NET}.mask.caffemodel
		;;
esac

# Start training.
time ./tools/train_net.py --gpu ${GPU_ID} \
	--solver output/${DATASET}/solver.prototxt \
	--weights ${NET_INIT} \
	--imdb "path" \
	--data-dir ${DATA_DIR} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml
