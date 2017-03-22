#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import argparse
import sys
import pprint
import numpy as np
# User-defined module
import _init_paths
from multitask_network_cascades.mnc_config import cfg, cfg_from_file, get_output_dir  # config mnc
from multitask_network_cascades.db.roidb import attach_roidb, prepare_roidb
from multitask_network_cascades.db.maskdb import attach_maskdb
from multitask_network_cascades.caffeWrapper.SolverWrapper import SolverWrapper
from multitask_network_cascades.datasets.path_db import PathDb
from multitask_network_cascades.mnc_config import cfg
import caffe


def parse_args():
    """ Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='path_db', type=str)
    parser.add_argument('--data-dir', dest='data_dir',
                        help='path to dataset to train on',
                        default='./data/VOCdevkitSDS', type=str)
    parser.add_argument('--image-set', dest='image_set',
                        help='image set to train on',
                        default='train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    if args.imdb_name == 'path':
        imdb   = PathDb(args.data_dir, args.image_set)
        roidb  = imdb.roidb
        maskdb = imdb.maskdb
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_rois()
            print('done')
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_masks()
            print('done')
        prepare_roidb(imdb)

    else:
        # get imdb and roidb from specified imdb_name
        imdb, roidb = attach_roidb(args.imdb_name, args.data_dir)

        # Faster RCNN doesn't need
        if cfg.MNC_MODE or cfg.CFM_MODE:
            imdb, maskdb = attach_maskdb(args.imdb_name, args.data_dir)
        else:
            maskdb = None
    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    _solver = SolverWrapper(args.solver, roidb, maskdb, output_dir, imdb,
                            pretrained_model=args.pretrained_model)

    print('Solving...')
    _solver.train_model(args.max_iters)
    print('done solving')

