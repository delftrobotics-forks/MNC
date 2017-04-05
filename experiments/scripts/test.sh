#!/bin/bash

GPU_ID=${1:-0}
NET=${2:-"ZF"}
STAGES=${3:-3}
DATA_DIR=${4:-"/srv/caffe-data/datasets/boxes_family_gray"}
ITERS=${5:-25000}
MODEL=${6:-"output/boxes_family_gray/zf_mnc_3stage_iter_25000.caffemodel.h5"}
TASK=${7:-"seg"}

read -p "Delete cache? [y/N] " yn
case $yn in
  [Yy]* ) rm -rf cache/*; [[ -d "output" ]] && find output -name "*.pkl" -type f -delete;
esac

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/mnc_${STAGES}stage/test.prototxt \
  --net ${MODEL} \
  --imdb "path" \
  --data-dir ${DATA_DIR} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml \
  --task ${TASK}
