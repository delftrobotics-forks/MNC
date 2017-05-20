# --------------------------------------------------------
# Multitask Network Cascade
# Modified by Mihai Morariu (m.a.morariu@delftrobotics.com)
# Copyright (c) 2017, Delft Robotics
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import json
import os

class MiscDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        top[0].reshape(1, 3, 2)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        im_path = "".join(map(chr, bottom[0].data))
        base_dir = os.path.dirname(im_path)
        filename = os.path.basename(im_path)
        json_path = os.path.join(base_dir, "../", "misc", os.path.splitext(filename)[0] + ".json")

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        top[0].reshape(len(json_data), 3, 2)
        for key, val in json_data.items():
            index = int(key) - 1
            top[0].data[index, 0, 0] = val["keypoints"]["left"]["x"]
            top[0].data[index, 0, 1] = val["keypoints"]["left"]["y"]
            top[0].data[index, 1, 0] = val["keypoints"]["hole"]["x"]
            top[0].data[index, 1, 1] = val["keypoints"]["hole"]["y"]
            top[0].data[index, 2, 0] = val["keypoints"]["right"]["x"]
            top[0].data[index, 2, 1] = val["keypoints"]["right"]["y"]

    def backward(self, top, propagate_down, bottom):
        pass
