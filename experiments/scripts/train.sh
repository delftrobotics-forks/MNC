#!/bin/bash

GPU_ID=${1:-0}
NET=${2:-"ZF"}
STAGES=${3:-3}
DATA_DIR=${4:-"/srv/caffe-data/datasets/boxes_family_gray"}
ITERS=${5:-25000}

read -p "Delete cache? [y/N] " yn
case $yn in
  [Yy]* ) rm -rf cache/*; rm -rf data/cache/*;
esac

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

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/mnc_${STAGES}stage/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb "path" \
  --data-dir ${DATA_DIR} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml
