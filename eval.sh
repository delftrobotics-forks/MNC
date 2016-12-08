rm data/VOCdevkitSDS/annotations_cache/*;
rm -r output/default;
rm -r data/cache/*;
python tools/test_net.py --gpu 0 --def /home/rmensing/dev/mnc/models/ResNet50/mnc_3stage/test.prototxt --net /home/rmensing/dev/mnc/output/mnc_3stage/voc_2012_train/resnet50_mnc_3stage_iter_25000.caffemodel.h5 --imdb voc_2012_seg_val --task seg --disparity
