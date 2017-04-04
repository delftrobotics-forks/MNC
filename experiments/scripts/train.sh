#!/bin/bash

DELETE_CACHE=${1:-true}
GPU_ID=${2:-0}
NET=${3:-"ZF"}
STAGES=${4:-3}
DATA_DIR=${5:-"/srv/caffe-data/datasets/boxes_family_gray"}
ITERS=${6:-25000}

[ "$DELETE_CACHE" = true ] && rm -rf cache/* && rm -rf data/cache/*

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
