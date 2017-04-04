#!/bin/bash

DELETE_CACHE=${1:-true}
GPU_ID=${2:-0}
NET=${3:-"ZF"}
STAGES=${4:-3}
DATA_DIR=${5:-"/srv/caffe-data/datasets/boxes_family_gray"}
ITERS=${6:-25000}
MODEL=${7:-"output/boxes_family_gray/zf_mnc_3stage_iter_25000.caffemodel.h5"}
TASK=${8:-"seg"}

if [ "$DELETE_CACHE" = true ]; then
  rm -rf cache/*
  [ -d "output" ] && find output -name "*.pkl" -type f -exec rm {} \;
fi

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/mnc_${STAGES}stage/test.prototxt \
  --net ${MODEL} \
  --imdb "path" \
  --data-dir ${DATA_DIR} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml \
  --task ${TASK}
