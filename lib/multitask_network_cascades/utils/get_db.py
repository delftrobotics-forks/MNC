#!/usr/bin/env python

from multitask_network_cascades.datasets.path_db import PathDb
from multitask_network_cascades.db.roidb import attach_roidb, prepare_roidb
from multitask_network_cascades.db.maskdb import attach_maskdb
from multitask_network_cascades.db.imdb import get_imdb
from multitask_network_cascades.mnc_config import cfg

import caffe

def get_db(imdb_name, data_dir, image_set, phase=caffe.TRAIN):
    if imdb_name == 'path':
        imdb   = PathDb(data_dir, image_set)
        if not phase == caffe.TRAIN:
            return imdb, None, None
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
        if phase == caffe.TRAIN:
            # get imdb and roidb from specified imdb_name
            imdb, roidb = attach_roidb(imdb_name, data_dir)

            # Faster RCNN doesn't need
            if cfg.MNC_MODE or cfg.CFM_MODE:
                imdb, maskdb = attach_maskdb(imdb_name, data_dir)
            else:
                maskdb = None
        else:
            return get_imdb(imdb_name), None, None
    return (imdb, roidb, maskdb)



