# -*- coding: utf-8 -*-

"""
python split_dataset.py --images_dir=data/images \
                        --annotations_dir=data/annotations \
                        --set_dir=data/set \
                        --rate=0.2
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

np.random.seed(23)

flags = tf.app.flags
flags.DEFINE_string('images_dir', '', '')
flags.DEFINE_string('annotations_dir', '', '')
flags.DEFINE_string('set_dir', '', '')
flags.DEFINE_float('rate', 0.2, '')

FLAGS = flags.FLAGS

def main(_):
    images_dir = FLAGS.images_dir
    annotations_dir = FLAGS.annotations_dir
    set_dir = FLAGS.set_dir
    rate = FLAGS.rate
    
    ann_files = np.random.permutation(os.listdir(annotations_dir))
    index = int(len(ann_files) * (1 -  rate))
    trn_ann_files = ann_files[: index]
    val_ann_files = ann_files[index:]
    
    with open(os.path.join(set_dir, 'train.txt'), 'wt') as f:
        for o in trn_ann_files:
            f.write(os.path.splitext(o)[0] + '\n')
    print('[INFO] `train.txt` saved in `{}` successfully'.format(set_dir))

    with open(os.path.join(set_dir, 'val.txt'), 'wt') as f:
        for o in val_ann_files:
            f.write(os.path.splitext(o)[0] + '\n')
    print('[INFO] `val.txt` saved in `{}` successfully'.format(set_dir))

if __name__ == '__main__':
    tf.app.run()