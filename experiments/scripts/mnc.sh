#!/bin/bash
# Usage:
# ./experiments/scripts/mnc.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/mnc.sh 0 VGG16 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
STAGES=$3
DATA_DIR=$4
NET_lc=${NET,,}
ITERS=25000
DATASET='path'
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/mnc_${STAGES}stage_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

case $NET in
	ZF)
		NET_INIT=data/imagenet_models/${NET}.caffemodel
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
  --imdb ${DATASET} \
  --data-dir ${DATA_DIR} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml \
  ${EXTRA_ARGS}
  
set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/mnc_${STAGES}stage/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET} \
  --data-dir ${DATA_DIR} \
  --cfg experiments/cfgs/${NET}/mnc_${STAGES}stage.yml \
  --task seg

